"""Line-aligned source ranges for live search follow-ups."""
import json
from functools import lru_cache
from pathlib import Path

from tree_sitter import Language, Parser, Query, QueryCursor
import tree_sitter_c
import tree_sitter_cpp

from .config import ConfigError


@lru_cache(maxsize=2)
def _grammar(cpp):
    language = Language((tree_sitter_cpp if cpp else tree_sitter_c).language())
    return language, Query(language, '(function_definition) @function')


def function_ranges(path, source):
    suffix = Path(path).suffix.lower()
    if suffix not in {'.c', '.h', '.cc', '.cpp', '.cxx', '.hpp', '.hh', '.hxx'}:
        return []
    language, query = _grammar(suffix != '.c')
    tree = Parser(language).parse(source.encode('utf-8'))
    nodes = QueryCursor(query).captures(tree.root_node).get('function', [])
    ranges = []
    for node in nodes:
        body = node.child_by_field_name('body')
        if body is None or body.type != 'compound_statement' or not body.child_count:
            continue
        closing = body.child(body.child_count - 1)
        if closing.type == '}' and not closing.is_missing:
            ranges.append((node.start_point.row + 1, closing.end_point.row + 1))
    return ranges


def read_context(ranges, start, end, count):
    enclosing = [(a, b) for a, b in ranges if a <= start and end <= b]
    if enclosing:
        a, b = min(enclosing, key=lambda span: span[1] - span[0])
        return 'function', a, b
    return 'surrounding_lines', max(1, start - 40), min(count, end + 40)


def pinned_read(value, start, end):
    locator = {key: value[key] for key in ('path', 'source_id', 'sha256')}
    return {'project': value['project'], 'requests': [
        {**locator, 'anchor_id': value['id'], 'start': start, 'end': end}], 'max_chars': 64000}


def page_end(value, lines, start, end, budget):
    """Fit complete lines, including space for the continuation when required."""
    if type(start) is not int or type(end) is not int or not 1 <= start <= end <= len(lines):
        raise ConfigError('Invalid source line range')
    meta = dict(value, text='', lines=[start, end], source_changed=False)
    envelope = {'state': 'ready', 'excerpts': [meta], 'truncated': False}
    def size(obj):
        return len(json.dumps(obj, ensure_ascii=False, separators=(',', ':')))
    room = budget - size(envelope)
    # Use the largest possible line number for the continuation to reserve enough space.
    page_room = room - size({'next_read': pinned_read(value, end, end)})
    used, stop = 0, start - 1
    for number in range(start, end + 1):
        used += size(lines[number - 1]) - 2  # surrounding JSON quotes are already in the envelope
        if used <= page_room:
            stop = number
        if used > room:
            if stop < start:
                raise ConfigError('Increase max_chars to fit a complete source line and its locator')
            return stop
    return end
