"""Static JavaScript/TypeScript syntax analysis. Never runs Node or project code."""
import posixpath
from collections import defaultdict
from pathlib import PurePosixPath

from tree_sitter import Language, Parser
import tree_sitter_javascript as javascript
import tree_sitter_typescript as typescript

JS_EXTENSIONS = {'.js', '.jsx', '.mjs', '.cjs', '.ts', '.tsx', '.mts', '.cts'}


def walk(root):
    stack = [root]
    while stack:
        node = stack.pop()
        yield node
        stack.extend(reversed(node.named_children))


class JavaScriptDependencies:
    def __init__(self, files):
        self.edges, self.definitions, self.spans = {}, {}, {}
        self.parse_failures = 0
        parsers = {
            'js': Parser(Language(javascript.language())),
            'ts': Parser(Language(typescript.language_typescript())),
            'tsx': Parser(Language(typescript.language_tsx())),
        }
        for name, text in files.items():
            suffix = PurePosixPath(name).suffix
            if suffix not in JS_EXTENSIONS:
                continue
            parser = parsers['tsx' if suffix == '.tsx' else 'ts' if suffix in {'.ts', '.mts', '.cts'} else 'js']
            raw = text.encode('utf-8')
            root = parser.parse(raw).root_node
            if root.has_error:
                self.parse_failures += 1
                continue
            found, definitions, spans, exported_aliases = defaultdict(set), {}, [], []
            for node in walk(root):
                if node.type in {'import_statement', 'export_statement'}:
                    source = node.child_by_field_name('source')
                    target = self.resolve(name, self.string(source), files)
                    if target:
                        symbols = [self.text(n.child_by_field_name('name')) for n in walk(node)
                                   if n.type in {'import_specifier', 'export_specifier'}]
                        clause = next((n for n in node.named_children if n.type == 'import_clause'), None)
                        if clause and any(n.type == 'identifier' for n in clause.named_children):
                            symbols.append('default')
                        found[target].update(symbols or [''])
                    if node.type == 'export_statement' and source is None:
                        for specifier in walk(node):
                            if specifier.type == 'export_specifier':
                                exported_aliases.append((self.text(specifier.child_by_field_name('alias')),
                                                         self.text(specifier.child_by_field_name('name'))))
                        value = node.child_by_field_name('value')
                        if value and value.type in {'arrow_function', 'function_expression', 'class'}:
                            start, end = node.start_point.row + 1, node.end_point.row + 1
                            definitions['default'] = start
                            spans.append({'name': 'default', 'start': start, 'end': end})
                        elif value and value.type == 'identifier':
                            exported_aliases.append(('default', self.text(value)))
                elif node.type == 'call_expression':
                    function = node.child_by_field_name('function')
                    args = node.child_by_field_name('arguments')
                    if function and function.type == 'identifier' and self.text(function) == 'require' and args:
                        values = args.named_children
                        if len(values) == 1:
                            target = self.resolve(name, self.string(values[0]), files)
                            if target:
                                found[target].add('')
                if node.type in {'function_declaration', 'generator_function_declaration',
                                  'class_declaration', 'method_definition', 'variable_declarator',
                                  'interface_declaration', 'type_alias_declaration'}:
                    identifier = node.child_by_field_name('name')
                    if identifier is None or identifier.type not in {'identifier', 'property_identifier', 'type_identifier'}:
                        continue
                    symbol = self.text(identifier)
                    start, end = node.start_point.row + 1, node.end_point.row + 1
                    definitions.setdefault(symbol, start)
                    value = node.child_by_field_name('value')
                    is_callable = node.type != 'variable_declarator' or (value and value.type in {
                        'arrow_function', 'function_expression', 'generator_function'})
                    if is_callable:
                        spans.append({'name': symbol, 'start': start, 'end': end})
                    parent = node.parent
                    if parent and parent.type == 'export_statement' and self.text(parent).lstrip().startswith('export default '):
                        definitions['default'] = start
            for exported, local in exported_aliases:
                if exported and local in definitions:
                    definitions[exported] = definitions[local]
            self.edges[name] = {target: sorted(symbols) for target, symbols in sorted(found.items()) if target != name}
            self.definitions[name], self.spans[name] = definitions, spans

    @staticmethod
    def text(node):
        return node.text.decode('utf-8') if node else ''

    @classmethod
    def string(cls, node):
        if not node or node.type != 'string':
            return ''
        text = cls.text(node)
        # Avoid inventing module paths when escape sequences need interpretation.
        return text[1:-1] if '\\' not in text else ''

    @staticmethod
    def resolve(importer, specifier, files):
        if not specifier.startswith(('./', '../')):
            return None
        stem = posixpath.normpath(posixpath.join(str(PurePosixPath(importer).parent), specifier))
        if stem == '..' or stem.startswith('../') or stem.startswith('/'):
            return None
        candidates = [stem]
        suffix = PurePosixPath(stem).suffix
        # TS projects commonly write .js import specifiers for .ts source files.
        substitutions = {'.js': ('.ts', '.tsx'), '.jsx': ('.tsx',), '.mjs': ('.mts',), '.cjs': ('.cts',)}
        candidates += [stem[:-len(suffix)] + ext for ext in substitutions.get(suffix, ())]
        if not suffix:
            candidates += [stem + ext for ext in ('.ts', '.tsx', '.js', '.jsx', '.mts', '.mjs', '.cts', '.cjs')]
            candidates += [stem + '/index' + ext for ext in ('.ts', '.tsx', '.js', '.jsx')]
        return next((candidate for candidate in candidates if candidate in files), None)
