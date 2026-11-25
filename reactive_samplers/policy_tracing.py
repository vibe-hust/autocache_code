"""Record policy conditions and return expressions without changing decisions."""
import ast


class TraceTests(ast.NodeTransformer):
    def visit_If(self, node):
        expression = ast.unparse(node.test)
        node = self.generic_visit(node)
        node.test = ast.Call(ast.Name('_guard', ast.Load()), [ast.Constant(expression), node.test], [])
        return node

    def visit_Compare(self, node):
        expression = ast.unparse(node)
        node = self.generic_visit(node)
        values = [node.left, *node.comparators]
        tagged = [ast.Call(ast.Name('_operand', ast.Load()), [ast.Constant(expression), ast.Constant(i), value], [])
                  for i, value in enumerate(values)]
        node.left, node.comparators = tagged[0], tagged[1:]
        return ast.copy_location(ast.Call(ast.Name('_test', ast.Load()), [ast.Constant(expression), node], []), node)

    def visit_Return(self, node):
        expression = ast.unparse(node.value)
        return ast.copy_location(ast.Return(ast.Call(ast.Name('_returned', ast.Load()),
            [ast.Constant(expression), self.visit(node.value)], [])), node)
