// Command instrument rewrites isolated source copies for serial payload metrics.
// Never run it on a production working tree or use traced builds for timings.
package main

import (
	"bytes"
	"fmt"
	"go/ast"
	"go/format"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

const traceImport = "github.com/itsmontoya/neuralnetwork/tools/nn13/trace"

func main() {
	if len(os.Args) != 2 {
		panic("usage: instrument SOURCE_ROOT")
	}

	if err := filepath.WalkDir(os.Args[1], instrumentFile); err != nil {
		panic(err)
	}
}

func instrumentFile(path string, entry os.DirEntry, walkErr error) (err error) {
	if walkErr != nil {
		return walkErr
	}

	if entry.IsDir() {
		if entry.Name() == ".git" || entry.Name() == "tools" || entry.Name() == "examples" || entry.Name() == "onnxpb" || entry.Name() == "testdata" {
			return filepath.SkipDir
		}

		return nil
	}

	if !strings.HasSuffix(path, ".go") || strings.HasSuffix(path, "_test.go") || strings.Contains(path, "metal") {
		return nil
	}

	var (
		file *ast.File
		data bytes.Buffer
	)
	positions := token.NewFileSet()
	if file, err = parser.ParseFile(positions, path, nil, parser.ParseComments); err != nil {
		return err
	}

	if file.Name.Name == "main" {
		return nil
	}

	changed := false
	for _, declaration := range file.Decls {
		function, ok := declaration.(*ast.FuncDecl)
		if !ok || function.Body == nil {
			continue
		}

		changed = true
		roots := names(function.Recv)
		roots = append(roots, names(function.Type.Params)...)
		roots = append(roots, names(function.Type.Results)...)
		ast.Inspect(function.Body, func(node ast.Node) (descend bool) {
			assignment, isAssignment := node.(*ast.AssignStmt)
			if function.Name.Name == "TransposeInto" && isAssignment && len(assignment.Rhs) == 1 {
				if _, indexed := assignment.Rhs[0].(*ast.IndexExpr); indexed {
					assignment.Rhs[0] = &ast.CallExpr{Fun: &ast.SelectorExpr{X: ast.NewIdent("nn13trace"), Sel: ast.NewIdent("ValueCopy")}, Args: []ast.Expr{assignment.Rhs[0]}}
				}
			}

			call, isCall := node.(*ast.CallExpr)
			if !isCall {
				return true
			}

			identifier, isIdentifier := call.Fun.(*ast.Ident)
			if !isIdentifier {
				return true
			}

			if identifier.Name == "copy" {
				call.Fun = &ast.SelectorExpr{X: ast.NewIdent("nn13trace"), Sel: ast.NewIdent("Copy")}
				return true
			}

			if identifier.Name != "make" || len(call.Args) < 2 {
				return true
			}

			slice, isSlice := call.Args[0].(*ast.ArrayType)
			if !isSlice || slice.Len != nil {
				return true
			}

			element, isElement := slice.Elt.(*ast.Ident)
			if !isElement {
				return true
			}

			numeric := element.Name == "float32"
			if element.Name == "int64" {
				numeric = (file.Name.Name == "tensor" && function.Name.Name == "Int64Values") || (file.Name.Name == "decode" && function.Name.Name == "convertTensor")
				if file.Name.Name == "tensor" && function.Name.Name == "newTensor" {
					// Shape and integer payload have distinct allocation lengths.
					numeric = !strings.Contains(expression(call.Args[1]), "len(shape)")
				}
			}

			if numeric {
				call.Fun = &ast.IndexExpr{X: &ast.SelectorExpr{X: ast.NewIdent("nn13trace"), Sel: ast.NewIdent("Make")}, Index: element}
				call.Args = call.Args[1:]
			}

			return true
		})
		prelude := fmt.Sprintf("package p;func f(){nn13trace.Enter(%s);defer func(){nn13trace.Leave(%s)}()}", strconv.Quote(file.Name.Name+"."+function.Name.Name), strings.Join(roots, ","))
		parsed, parseErr := parser.ParseFile(positions, "prelude.go", prelude, 0)
		if parseErr != nil {
			return parseErr
		}

		statements := parsed.Decls[0].(*ast.FuncDecl).Body.List
		function.Body.List = append(statements, function.Body.List...)
	}
	if !changed {
		return nil
	}

	file.Decls = append([]ast.Decl{&ast.GenDecl{Tok: token.IMPORT, Specs: []ast.Spec{&ast.ImportSpec{Name: ast.NewIdent("nn13trace"), Path: &ast.BasicLit{Kind: token.STRING, Value: strconv.Quote(traceImport)}}}}}, file.Decls...)
	if err = format.Node(&data, positions, file); err != nil {
		return err
	}

	if err = os.WriteFile(path, data.Bytes(), 0600); err != nil {
		return fmt.Errorf("instrument %s: %w", path, err)
	}

	return nil
}

func names(fields *ast.FieldList) (out []string) {
	if fields == nil {
		return nil
	}

	for _, field := range fields.List {
		for _, name := range field.Names {
			if name.Name != "_" {
				out = append(out, name.Name)
			}
		}
	}
	return out
}

func expression(node ast.Node) (text string) {
	var output bytes.Buffer
	if err := format.Node(&output, token.NewFileSet(), node); err != nil {
		panic(err)
	}

	return output.String()
}
