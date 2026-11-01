"""Static Python import resolution over already indexed local files only."""
import ast
from collections import defaultdict
from pathlib import PurePosixPath


class PythonDependencies:
    def __init__(self, files):
        self.edges = {}
        self.definitions = {}
        self.spans = {}
        self.parse_failures = 0
        self.files = files
        for name, text in files.items():
            try:
                tree = ast.parse(text, filename=name)
            except (SyntaxError, ValueError, RecursionError):
                self.parse_failures += 1
                continue
            definitions = {}
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    definitions[node.name] = node.lineno
                elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                    for target in targets:
                        if isinstance(target, ast.Name):
                            definitions[target.id] = node.lineno
            self.definitions[name] = definitions
            self.spans[name] = [
                {"name": node.name, "start": min([node.lineno] +
                 [d.lineno for d in node.decorator_list]), "end": node.end_lineno}
                for node in ast.walk(tree)
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            ]
            found = defaultdict(set)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        self._add(found, self._absolute(alias.name, name), None)
                elif isinstance(node, ast.ImportFrom):
                    if node.level:
                        parent = list(PurePosixPath(name).parent.parts)
                        if node.level > len(parent):
                            continue
                        base = parent[:len(parent) - node.level + 1]
                        if node.module:
                            base += node.module.split('.')
                        stem = '/'.join(base)
                        target = self._resolve(stem)
                        for alias in node.names:
                            child = self._resolve(stem + '/' + alias.name)
                            self._add(found, child or target, None if child else alias.name)
                    else:
                        module = node.module or ''
                        target = self._absolute(module, name)
                        for alias in node.names:
                            child = self._absolute(module + '.' + alias.name, name)
                            self._add(found, child or target, None if child else alias.name)
            # Package initializers can also define import-time behavior.
            for target in list(found):
                parts = PurePosixPath(target).parent.parts
                for length in range(1, len(parts) + 1):
                    initializer = '/'.join(parts[:length]) + '/__init__.py'
                    if initializer in files and initializer != name:
                        found[initializer].add('')
            self.edges[name] = {target: sorted(symbols) for target, symbols in sorted(found.items())
                                if target != name}

    def _resolve(self, stem):
        for suffix in ('.py', '/__init__.py', '.pyi', '/__init__.pyi'):
            if stem + suffix in self.files:
                return stem + suffix
        return None

    def _absolute(self, module, importer):
        stem = module.replace('.', '/')
        for prefix in ('', 'src/', str(PurePosixPath(importer).parent) + '/'):
            result = self._resolve(prefix + stem)
            if result:
                return result
        return None

    @staticmethod
    def _add(found, target, symbol):
        if target:
            found[target].add('' if symbol in (None, '*') else symbol)
