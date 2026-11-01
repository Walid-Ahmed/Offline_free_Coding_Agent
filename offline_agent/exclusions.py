"""Root-scoped Git-style exclusion rules, with no Git subprocess or network."""
from pathlib import Path

from pathspec import GitIgnoreSpec


class IgnoreConfigurationError(ValueError):
    pass


class Exclusions:
    def __init__(self, root, patterns=()):
        self.root = root
        self.patterns = tuple(patterns)
        self.cache = {}
        for pattern in self.patterns:
            if not isinstance(pattern, str) or not pattern.strip() or '\n' in pattern or '\r' in pattern:
                raise IgnoreConfigurationError("--exclude needs a nonblank single-line Git-style pattern")
            if pattern.startswith('!'):
                raise IgnoreConfigurationError("--exclude is deny-only; use .agentignore for ordered negation rules")
        self.extra = self.compile(self.patterns, '--exclude')
        for filename in ('.gitignore', '.agentignore'):
            self.spec(Path(), filename)

    @staticmethod
    def compile(lines, source):
        try:
            return GitIgnoreSpec.from_lines(lines)
        except Exception as exc:
            raise IgnoreConfigurationError(f"Invalid exclusion rules in {source}: {exc}") from exc

    def spec(self, directory, filename):
        key = (directory, filename)
        if key not in self.cache:
            path = self.root / directory / filename
            if path.is_symlink():
                self.cache[key] = None  # Git does not follow symlinked .gitignore files.
            else:
                try:
                    if path.exists() and not path.is_file():
                        raise IgnoreConfigurationError(f"Exclusion configuration must be a regular file: {path}")
                    with path.open('rb') as handle:
                        data = handle.read(64_001)
                    if len(data) > 64_000:
                        raise IgnoreConfigurationError(f"Exclusion file exceeds 64 KB: {path}")
                    self.cache[key] = self.compile(data.decode('utf-8').splitlines(), path)
                except FileNotFoundError:
                    self.cache[key] = None
                except (OSError, UnicodeError) as exc:
                    raise IgnoreConfigurationError(f"Cannot read exclusion rules in {path}: {exc}") from exc
        return self.cache[key]

    def layer(self, relative, directory, filename):
        ignored = False
        for depth in range(len(relative.parts)):
            ancestor = Path(*relative.parts[:depth])
            spec = self.spec(ancestor, filename)
            if spec is not None:
                local = Path(*relative.parts[depth:]).as_posix() + ('/' if directory else '')
                decision = spec.check_file(local).include
                if decision is not None:
                    ignored = decision
        return ignored

    def ignored(self, relative, directory=False):
        # Evaluate parents too: a child cannot re-include an excluded directory.
        for depth in range(1, len(relative.parts) + 1):
            prefix = Path(*relative.parts[:depth])
            is_dir = depth < len(relative.parts) or directory
            if (self.layer(prefix, is_dir, '.gitignore')
                    or self.layer(prefix, is_dir, '.agentignore')
                    or self.extra.match_file(prefix.as_posix() + ('/' if is_dir else ''))):
                return True
        return False
