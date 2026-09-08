"""Publication boundary for documentation projections; never print matched values."""
from __future__ import annotations

import ast
from collections import defaultdict
from functools import lru_cache
import hashlib
import html
from html.parser import HTMLParser
import json
from pathlib import Path
import re
from urllib.parse import parse_qsl, unquote, urlsplit

REDACTED = "[redacted]"
# These identify private operational material, not public documentation domains.
PATTERNS = (
    re.compile(r"-----BEGIN [^-]*PRIVATE KEY-----.*?-----END [^-]*PRIVATE KEY-----", re.S),
    re.compile(r"\b(?:sk-(?:proj-)?|gh[pousr]_|github_pat_)[A-Za-z0-9_-]{8,}\b"),
    re.compile(r"\bAKIA[A-Z0-9]{16}\b"),
    re.compile(r"\bGPU-[0-9a-fA-F-]{8,}\b"),
    re.compile(r"(?<![\w:/.])(?:[A-Za-z]:[\\/]|/)[\w.@+~\\/-]*[\w.@+~-]"),
    re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b"),
    re.compile(r"\b(?:th-cl|kng-cl|lxplus|login|compute|worker|node)[-_]?[A-Za-z]*\d+[\w.-]*\b", re.I),
    re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    re.compile(r"\b(?:boyang(?:\.yu|yu)?|Boyang\.Yu)\b", re.I),
    re.compile(r"(?i)\b(?:job(?:[_ -]?id)?|slurm[_ -]?job[_ -]?id|scheduler[_ -]?id)\s*[:=#]?\s*[\"']?\d{4,}"),
    re.compile(r"(?i)\b(?:password|passwd|api[_ -]?key|secret|access[_ -]?token|auth[_ -]?token|bearer|username|hostname|gpu_uuid)\s*[:=]\s*[\"']?[^\s,;\"'}]+"),
    re.compile(r"(?i)\b(?:artifacts|checkpoints?|authority|authorities)/[^\s\"'<>`]+"),
    re.compile(r"(?i)\b[A-Za-z_][A-Za-z0-9_-]*\.(?:pt|pth|ckpt)\b"),
    re.compile(r"(?i)\b[\w-]+(?:\.[\w-]+)*\.(?:internal|local|desy\.de|cern\.ch)\b"),
    re.compile(r"(?i)\bfile://[^\s<>\"'`]+"),
    re.compile(r"\\\\[^\\\s]+\\[^\s<>\"'`]+"),
)
# Each lower-case marker tuple is a necessary (never merely likely) condition
# for the corresponding expression above. This avoids repeatedly running
# expensive regular expressions over multi-megabyte generated pages while
# preserving the exact fail-closed match set.
_PATTERN_MARKERS = (
    ("private key",),
    ("sk-", "ghp_", "gho_", "ghu_", "ghs_", "ghr_", "github_pat_"),
    ("akia",),
    ("gpu-",),
    ("/", "\\"),
    ("@",),
    ("th-cl", "kng-cl", "lxplus", "login", "compute", "worker", "node"),
    (".",),
    ("boyang",),
    ("job", "slurm", "scheduler"),
    ("password", "passwd", "api", "secret", "access", "auth", "bearer",
     "username", "hostname", "gpu"),
    ("artifact", "checkpoint", "authorit"),
    (".pt", ".ckpt"),
    (".internal", ".local", ".desy.de", ".cern.ch"),
    ("file://",),
    ("\\\\",),
)

PUBLIC_HOSTS = frozenset({
    "software.belle2.org", "docs.github.com", "github.com", "www.sphinx-doc.org",
    "www.belle2.org", "docs.python.org", "pytorch.org", "numpy.org", "scipy.org",
    "docs.scipy.org", "matplotlib.org", "onnxruntime.ai", "onnx.ai", "pypi.org",
    "packaging.python.org", "developer.mozilla.org", "www.w3.org",
})
_URL = re.compile(r"https?://[^\s<>\"'`]+", re.I)
_SENSITIVE_KEY = re.compile(
    r"(?:^|_)(?:password|passwd|secret|secrets|credentials?|api_key|access_token|"
    r"auth_token|bearer|username|user_name|hostname|host_name|host(?![_-])|"
    r"gpu_uuid|job_id|scheduler_id|authority_path|checkpoint_path)(?:$|_)", re.I,
)
_EMPTY_PRIVATE_VALUES = {
    "", "none", "null", "true", "false", "unknown", "unavailable",
    "not_run", "not_reviewed", "read", "write", REDACTED,
}
_PRIVATE_URL_PATH = re.compile(
    r"(?:^|/)(?:home|users|scratch|gpfs|lustre|cvmfs|mnt|tmp)(?:/|$)", re.I,
)
_CSS_ESCAPE = re.compile(r"\\([0-9a-fA-F]{1,6})(?:[ \t\r\n\f])?")
_JS_OCTAL_ESCAPE = re.compile(r"\\([0-7]{1,3})")
_SURROGATE_PAIR = re.compile(
    f"([{chr(0xD800)}-{chr(0xDBFF)}])"
    f"([{chr(0xDC00)}-{chr(0xDFFF)}])"
)
_SURROGATE = re.compile(f"[{chr(0xD800)}-{chr(0xDFFF)}]")
_CSS_CONTINUATION = re.compile(r"\\(?:\r\n|[\n\r\f])")
_CSS_SIMPLE_ESCAPE = re.compile(r"\\([^\n\r\f0-9a-fA-F])")
_FIELD_ASSIGNMENT = re.compile(
    r'''(?ix)(?<![\w])
        (?P<key_quote>["']?)(?P<key>
            authorization|auth[_-]?header|auth|token|cookie|session|session[_-]?id|
            private[_-]?key(?:[_-]?id)?|ssh[_-]?key(?:[_-]?path)?|
            access[_-]?key(?:[_-]?id)?|secret[_-]?key|auth[_-]?key|signing[_-]?key|
            encryption[_-]?key|password|passwd|passphrase|db[_-]?pass|pwd|
            secrets?|credentials?|
            api[_-]?key|access[_-]?token|auth[_-]?token|bearer|
            user(?:name|[_-]name)|host(?:name|[_-]name)|host(?![_-])|gpu[_-]?uuid|
            job[_-]?id|scheduler[_-]?id|authority[_-]?path|checkpoint[_-]?path|
            [A-Za-z_][A-Za-z0-9_-]*(?:token|private[_-]?key|ssh[_-]?key|cookie|
                secret[_-]?key|auth[_-]?key|signing[_-]?key|encryption[_-]?key|
                password|passwd|passphrase|db[_-]?pass|secrets?|credentials?|
                username|hostname)
            |[A-Za-z_][A-Za-z0-9_-]*(?:authorization|auth[_-]?header|auth|
                bearer|session[_-]?id|session|host|access[_-]?key(?:[_-]?id)?|
                private[_-]?key[_-]?id|ssh[_-]?key[_-]?path)
        )(?P=key_quote)\s*(?P<delimiter>[:=])\s*
        (?P<value>"(?:\\.|[^"\\])+"|'(?:\\.|[^'\\])+'|[^\s,;}\]]+)
    ''',
)
# Backreferences make ``_FIELD_ASSIGNMENT`` unsuitable for the bounded
# Thompson automaton used across HTML parser channels.  This conservative
# equivalent intentionally permits mismatched optional key quotes while
# retaining the same sensitive-key families and requiring a non-empty value.
_ORDERED_FIELD_ASSIGNMENT = re.compile(
    r'''(?ix)(?<![\w])["']?(?:
        authorization|auth[_-]?header|auth|token|cookie|session|session[_-]?id|
        private[_-]?key(?:[_-]?id)?|ssh[_-]?key(?:[_-]?path)?|
        access[_-]?key(?:[_-]?id)?|secret[_-]?key|auth[_-]?key|signing[_-]?key|
        encryption[_-]?key|password|passwd|passphrase|db[_-]?pass|pwd|
        secrets?|credentials?|
        api[_-]?key|access[_-]?token|auth[_-]?token|bearer|
        user(?:name|[_-]name)|host(?:name|[_-]name)|host(?![_-])|gpu[_-]?uuid|
        job[_-]?id|scheduler[_-]?id|authority[_-]?path|checkpoint[_-]?path|
        [A-Za-z_][A-Za-z0-9_-]*(?:token|private[_-]?key|ssh[_-]?key|cookie|
            secret[_-]?key|auth[_-]?key|signing[_-]?key|encryption[_-]?key|
            password|passwd|passphrase|db[_-]?pass|secrets?|credentials?|
            username|hostname)
        |[A-Za-z_][A-Za-z0-9_-]*(?:authorization|auth[_-]?header|auth|
            bearer|session[_-]?id|session|host|access[_-]?key(?:[_-]?id)?|
            private[_-]?key[_-]?id|ssh[_-]?key[_-]?path)
    )["']?\s*[:=]\s*["']?[^\s,;}\]"']+
    ''',
)
_TYPE_ANNOTATION_VALUES = frozenset({
    "Any", "None", "bool", "bytes", "float", "int", "str", "Path",
    "Mapping", "MutableMapping", "Sequence", "Iterable", "Iterator",
    "Callable", "dict", "list", "tuple", "set", "frozenset",
    "torch.Tensor", "numpy.ndarray", "np.ndarray",
})
_ACTIVE_SCHEME = re.compile(r"(?<![A-Za-z0-9_+.-])(?:data|javascript)\s*:", re.I)
_REMOTE_CSS_URL = re.compile(r"(?:https?:)?//|https?:", re.I)
_CSS_CONTENT = re.compile(r"(?:^|[;{}])\s*content\s*:\s*([^;}]*)(?=[;}]|$)", re.I)
_CSS_GENERATED_TEXT = re.compile(
    r"@counter-style\b|\bsymbols\s*\(|"
    r"\blist-style(?:-type)?\s*:\s*(?:symbols\s*\(|[\"'])|"
    r"\bquotes\s*:",
    re.I,
)
_PRIVATE_KEY_HEADER = re.compile(r"-----BEGIN [^-]*PRIVATE KEY-----", re.I)
_DISTINCTIVE_ORDERED_PATTERNS = (0, 1, 2, 3, 8)
_BROAD_ORDERED_PATTERNS = tuple(
    index for index in range(len(PATTERNS))
    if index not in _DISTINCTIVE_ORDERED_PATTERNS
)
_SUBSTRING_ORDERED_PATTERNS = (5, 6, 7, 9, 13)
_ABSOLUTE_PATH_ORDERED_PATTERNS = (4,)
_PATH_FRAGMENT_ORDERED_PATTERNS = tuple(
    index for index in _BROAD_ORDERED_PATTERNS
    if index not in _SUBSTRING_ORDERED_PATTERNS
    and index not in _ABSOLUTE_PATH_ORDERED_PATTERNS
    and index != 10
)
_MAX_PUBLICATION_ENTRIES = 5_000
_MAX_PUBLICATION_FILE_BYTES = 10 * 1024 * 1024
_MAX_PUBLICATION_BYTES = 64 * 1024 * 1024
_MAX_SOURCE_TEXT_BYTES = 5_000_000
_MAX_TYPE_ANNOTATION_BYTES = 8_192
_MAX_MATCH_STATES = 100_000
_MAX_MATCH_OPERATIONS = 30_000_000
_MAX_ENCODED_STATES = 4_096
# A publication contains several independently meaningful parser projections.
# Ten per-file windows keep the 64 MiB site bounded while allowing the
# complete generated API inventory and HTML tree to be checked in one pass.
_MAX_PUBLICATION_MATCH_OPERATIONS = 300_000_000
_SAFE_CSS_CONTENT_VALUES = frozenset({"''", '\"\"', "':'", '\":\"', "'['", '\"[\"', "']'", '\"]\"'})
_BINARY_ASSETS = {".png", ".gif", ".jpg", ".jpeg", ".ico", ".woff", ".woff2", ".ttf", ".eot"}
_SAFE_INLINE_SCRIPTS = {
    'document.getElementById(\'searchbox\').style.display = "block"',
    "$('#searchbox').show(0);",
    "$('#fallback').hide();",
}
_PUBLIC_HTML_RECONSTRUCTION_RESETS = frozenset({
    "viewport",
    "width=device-width, initial-scale=1",
    "width=device-width, initial-scale=1.0",
})
_SAFE_GENERATED_JS = {
    # Sphinx 3.4 and Sphinx 7.3+ documentation_options.js for this fixed
    # project configuration. Dynamic search data is validated separately.
    "documentation_options.js": {
        "7db0e507691fe953e946705422fc34700e245f238748772799b13a96c6d93a11",
        "72cf26281405e94894f511652684c36cc57e6e5668a0b6576a22844ba55ef5f2",
    },
    "sidebar.js": {
        "73aad3bd47b163f696fc4f1eb942293f818bab54c34f60c9042a191bfd3ab5d0",
        "500d8aab35fc2c64962c513aa3d305328635b5e28f59a9a704b86c4ed8778cec",
        "c369ea07e39afac090a9d3e23921217a83d8aab4e20d27dc847b71807da44e63",
    },
}

# Parser projections deliberately concatenate semantic channels that a browser
# or index consumer can recombine. Word boundaries in the source-text rules
# are therefore not meaningful on those synthetic streams: an unrelated
# preceding key/name must not hide a credential or private identifier. The raw
# payload still uses the original boundary-aware expressions.
_BOUNDARYLESS_SEMANTIC_PATTERNS = frozenset({0, 1, 2, 3, 8})
_SEMANTIC_PATTERNS = tuple(
    re.compile(
        (
            pattern.pattern.replace(r"\b", "")
            if index in _BOUNDARYLESS_SEMANTIC_PATTERNS
            # ``.pt`` is too short for cross-element reconstruction: ordinary
            # Sphinx viewport metadata can synthesize it. It remains blocked
            # whenever contiguous; longer checkpoint suffixes stay ordered.
            else pattern.pattern.replace("(?:pt|pth|ckpt)", "(?:pth|ckpt)")
            if index == 12 else pattern.pattern
        ),
        pattern.flags,
    )
    for index, pattern in enumerate(PATTERNS)
)
_RUNTIME_URL_ATTRIBUTES = frozenset({
    ("script", "src"), ("link", "href"), ("img", "src"),
    ("iframe", "src"), ("frame", "src"), ("source", "src"),
    ("body", "background"), ("table", "background"),
    ("colgroup", "background"), ("col", "background"),
    ("thead", "background"), ("tbody", "background"),
    ("tfoot", "background"), ("tr", "background"),
    ("td", "background"), ("th", "background"),
    ("html", "manifest"), ("a", "ping"), ("area", "ping"),
    ("video", "src"),
    ("video", "poster"), ("audio", "src"), ("track", "src"),
    ("embed", "src"), ("object", "data"), ("input", "src"),
    ("image", "href"), ("image", "xlink:href"),
    ("use", "href"), ("use", "xlink:href"),
})
_STANDARD_HTML_ELEMENTS = frozenset({
    "a", "abbr", "address", "area", "article", "aside", "audio", "b",
    "base", "bdi", "bdo", "blockquote", "body", "br", "button", "canvas",
    "caption", "cite", "code", "col", "colgroup", "data", "datalist", "dd",
    "del", "details", "dfn", "dialog", "div", "dl", "dt", "em", "embed",
    "fieldset", "figcaption", "figure", "footer", "form", "h1", "h2", "h3",
    "h4", "h5", "h6", "head", "header", "hgroup", "hr", "html", "i",
    "iframe", "img", "input", "ins", "kbd", "label", "legend", "li",
    "link", "main", "map", "mark", "menu", "meta", "meter", "nav",
    "noscript", "object", "ol", "optgroup", "option", "output", "p",
    "picture", "pre", "progress", "q", "rp", "rt", "ruby", "s", "samp",
    "script", "search", "section", "select", "slot", "small", "source",
    "span", "strong", "style", "sub", "summary", "sup", "svg", "table",
    "tbody", "td", "template", "textarea", "tfoot", "th", "thead", "time",
    "title", "tr", "track", "u", "ul", "use", "var", "video", "wbr",
})
_STANDARD_HTML_ATTRIBUTES = frozenset({
    "accept", "accept-charset", "accesskey", "action", "align", "allow",
    "alt", "as", "async", "autocomplete", "autofocus", "autoplay",
    "background", "charset", "checked", "cite", "class", "cols", "colspan",
    "content", "contenteditable", "controls", "coords", "crossorigin", "data",
    "datetime", "decoding", "default", "defer", "dir", "dirname", "disabled",
    "download", "draggable", "enctype", "enterkeyhint", "for", "form",
    "formaction", "formenctype", "formmethod", "formnovalidate", "formtarget",
    "headers", "height", "hidden", "high", "href", "hreflang", "http-equiv",
    "id", "imagesizes", "imagesrcset", "inert", "inputmode", "integrity", "is",
    "ismap", "itemid", "itemprop", "itemref", "itemscope", "itemtype", "kind",
    "label", "lang", "list", "loading", "loop", "low", "manifest", "max",
    "maxlength", "media", "method", "min", "minlength", "multiple", "muted",
    "name", "nonce", "novalidate", "open", "optimum", "pattern", "ping",
    "placeholder", "playsinline", "poster", "preload", "readonly", "referrerpolicy",
    "rel", "required", "reversed", "role", "rows", "rowspan", "sandbox",
    "scope", "selected", "shape", "size", "sizes", "slot", "span", "spellcheck",
    "src", "srcdoc", "srclang", "srcset", "start", "step", "style", "tabindex",
    "target", "title", "translate", "type", "usemap", "value", "width", "wrap",
    "xlink:href",
})
_NONVISIBLE_HTML_ELEMENTS = frozenset({
    "dialog", "head", "template", "script", "style", "noscript",
    "title", "datalist",
})
_VOID_HTML_ELEMENTS = frozenset({
    "area", "base", "br", "col", "embed", "hr", "img", "input", "link",
    "meta", "source", "track", "wbr",
})


class _UnsafeEncoding(ValueError):
    """An encoded value exceeded the bounded, fail-closed decoder."""


class _DuplicateStructuredKey(ValueError):
    """A structured source would discard an earlier mapping value."""


class _PublicationMatchLimit(ValueError):
    """A bounded publication reconstruction exhausted its operation budget."""


def _decoded(text: str) -> str:
    """Normalize common HTML, URL and JavaScript escapes without evaluation."""
    result = str(text)
    if not any(marker in result for marker in ("&", "%", "\\")):
        return result
    for _ in range(32):
        previous = result
        result = unquote(html.unescape(result)).replace(r"\/", "/")
        result = re.sub(r"\\u\{([0-9a-fA-F]{1,6})\}|\\u([0-9a-fA-F]{4})|\\x([0-9a-fA-F]{2})",
                        lambda match: chr(int(next(group for group in match.groups() if group is not None), 16))
                        if int(next(group for group in match.groups() if group is not None), 16) <= 0x10FFFF else REDACTED,
                        result)
        result = _SURROGATE_PAIR.sub(
            lambda match: chr(
                0x10000
                + (ord(match.group(1)) - 0xD800) * 0x400
                + ord(match.group(2)) - 0xDC00
            ),
            result,
        )
        if _SURROGATE.search(result):
            raise _UnsafeEncoding("Invalid JavaScript Unicode surrogate")
        if result == previous:
            return result
    raise _UnsafeEncoding("Excessive nested encoding")


def _decoded_variants(text: str) -> tuple[str, ...]:
    """Return common, CSS-hex and JavaScript-octal decoded interpretations."""
    base = _decoded(text)
    if "\\" not in base:
        return (base,)
    results = [base]
    for escape in (_CSS_ESCAPE, _JS_OCTAL_ESCAPE):
        result = base
        for _ in range(32):
            previous = result
            result = escape.sub(
                lambda match: chr(int(match.group(1), 16 if escape is _CSS_ESCAPE else 8)),
                result,
            )
            if escape is _CSS_ESCAPE:
                result = _CSS_CONTINUATION.sub("", result)
                result = _CSS_SIMPLE_ESCAPE.sub(r"\1", result)
            result = _decoded(result)
            if result == previous:
                break
        else:
            raise _UnsafeEncoding("Excessive nested encoding")
        if result not in results:
            results.append(result)
    return tuple(results)


@lru_cache(maxsize=1)
def _html_entity_prefixes() -> frozenset[str]:
    """Return bounded prefixes of the standard HTML named-reference table."""
    return frozenset(
        name[:width]
        for name in html.entities.html5
        for width in range(1, min(32, len(name.rstrip(";"))) + 1)
    )


def _incomplete_encoded_suffixes(value: str) -> frozenset[str]:
    """Return only trailing escape prefixes that another token can complete.

    A literal percent (notably ``width: 100%``) and a prose ampersand are not
    encodings. Numeric HTML references are unambiguous; named references are
    decoded when complete inside one parser token and otherwise remain prose.
    """
    result = set()
    percent = re.search(r"%(?:[0-9A-Fa-f])?$", value)
    if percent is not None:
        result.add(percent.group(0))
    percent_bytes = re.search(r"(?:%[0-9A-Fa-f]{2}){1,3}$", value)
    if percent_bytes is not None:
        encoded = percent_bytes.group(0)
        raw = bytes(
            int(encoded[index + 1:index + 3], 16)
            for index in range(0, len(encoded), 3)
        )
        try:
            raw.decode("utf-8")
        except UnicodeDecodeError as error:
            if error.reason == "unexpected end of data" and error.end == len(raw):
                result.add(encoded)
    # Numeric references outside Unicode's representable range cannot decode
    # into useful publication text. Bound retained prefixes to that range.
    entity = re.search(
        r"&#(?:[xX][0-9A-Fa-f]{0,5}|[0-9]{0,6})$", value,
    )
    if entity is not None:
        result.add(entity.group(0))
    named_entity = re.search(r"&[A-Za-z][A-Za-z0-9]{0,31}$", value)
    if named_entity is not None:
        prefix = named_entity.group(0)
        if prefix[1:] in _html_entity_prefixes():
            result.add(prefix)
    elif value.endswith("&"):
        result.add("&")
    escape = re.search(
        r"\\(?:[uU](?:[0-9A-Fa-f]{0,3}|\{[0-9A-Fa-f]{0,6})|"
        r"[xX][0-9A-Fa-f]?|[0-9A-Fa-f]{1,5})$",
        value,
    )
    if escape is not None:
        result.add(escape.group(0))
    elif value.endswith("\\"):
        result.add("\\")
    return frozenset(result)


def _encoded_prefix_completion_length(
    prefix: str, continuation: str,
) -> int | None:
    """Return continuation characters consumed by one completed escape."""
    candidate = prefix + continuation

    def percent_utf8_tail(encoded_prefix: str, remaining: str) -> int | None:
        """Consume percent bytes until an incomplete UTF-8 scalar is whole."""
        encoded = encoded_prefix
        consumed = 0
        for _ in range(3):
            raw = bytes(
                int(encoded[index + 1:index + 3], 16)
                for index in range(0, len(encoded), 3)
            )
            try:
                raw.decode("utf-8")
            except UnicodeDecodeError as error:
                if error.reason != "unexpected end of data" or error.end != len(raw):
                    return None
            else:
                return consumed
            match = re.match(r"%[0-9A-Fa-f]{2}", remaining[consumed:])
            if match is None:
                return None
            encoded += match.group(0)
            consumed += match.end()
        return None

    if re.fullmatch(r"(?:%[0-9A-Fa-f]{2}){1,3}", prefix):
        consumed = percent_utf8_tail(prefix, continuation)
        return consumed if consumed is not None and consumed > 0 else None
    if prefix.startswith("%"):
        match = re.match(r"%[0-9A-Fa-f]{2}", candidate)
        consumed = None if match is None else match.end() - len(prefix)
        if consumed is not None and consumed > 0:
            encoded = candidate[:match.end()]
            tail = percent_utf8_tail(encoded, continuation[consumed:])
            if tail is not None:
                consumed += tail
        return consumed if consumed is not None and consumed > 0 else None
    if prefix.lower().startswith("&#x"):
        match = re.match(
            r"&#[xX][0-9A-Fa-f]{1,6}(?:;|(?=[^0-9A-Fa-f]))",
            candidate,
        )
        consumed = None if match is None else match.end() - len(prefix)
        return consumed if consumed is not None and consumed > 0 else None
    if prefix.startswith("&#"):
        match = re.match(
            r"&#[0-9]{1,7}(?:;|(?=[^0-9]))", candidate,
        )
        consumed = None if match is None else match.end() - len(prefix)
        return consumed if consumed is not None and consumed > 0 else None
    if prefix == "&" and continuation.startswith("#"):
        match = re.match(
            r"&#(?:[xX][0-9A-Fa-f]{1,6}|[0-9]{1,7})(?:;|(?=[^0-9A-Fa-f]))",
            candidate,
        )
        consumed = None if match is None else match.end() - len(prefix)
        return consumed if consumed is not None and consumed > 0 else None
    if prefix.startswith("&"):
        match = re.match(r"&[A-Za-z][A-Za-z0-9]*;", candidate)
        if match is None or match.group(0)[1:] not in html.entities.html5:
            return None
        consumed = match.end() - len(prefix)
        return consumed if consumed > 0 else None
    if prefix[:2].lower() == r"\u":
        if prefix[2:3] == "{":
            match = re.match(r"\\[uU]\{[0-9A-Fa-f]{1,6}\}", candidate)
        else:
            match = re.match(r"\\[uU][0-9A-Fa-f]{4}", candidate)
        consumed = None if match is None else match.end() - len(prefix)
        return consumed if consumed is not None and consumed > 0 else None
    if prefix[:2].lower() == r"\x":
        match = re.match(r"\\[xX][0-9A-Fa-f]{2}", candidate)
        consumed = None if match is None else match.end() - len(prefix)
        return consumed if consumed is not None and consumed > 0 else None
    # CSS consumes up to six hexadecimal characters and an optional spacing
    # terminator. A bare backslash can also escape one following character.
    if prefix == "\\":
        if re.match(r"[xX][0-9A-Fa-f]{2}", continuation):
            return 3
        if re.match(r"[uU][0-9A-Fa-f]{4}", continuation):
            return 5
        if re.match(r"[uU]\{[0-9A-Fa-f]{1,6}\}", continuation):
            match = re.match(r"[uU]\{[0-9A-Fa-f]{1,6}\}", continuation)
            return match.end() if match is not None else None
        match = re.match(
            r"[0-9A-Fa-f]{1,6}(?:\s|(?=[^0-9A-Fa-f])|$)",
            continuation,
        )
        if match is not None:
            return match.end()
        return None
    match = re.match(
        r"\\[0-9A-Fa-f]{1,6}(?:\s|(?=[^0-9A-Fa-f]))", candidate,
    )
    consumed = None if match is None else match.end() - len(prefix)
    return consumed if consumed is not None and consumed > 0 else None


def _has_useful_cross_fragment_encoding(
    tokens: list[tuple], publication_budget: list[int], file_budget: list[int],
) -> bool:
    """Reject cross-token escapes that decode to selectable ASCII or spacing."""
    # Cross-token escape syntax is not emitted by Sphinx. Track only the bounded
    # escape prefix, never an exponentially growing set of full carriers. A
    # completed replacement character is harmless (for example an accidental
    # ``%`` followed later by ``da``); printable ASCII, spacing, or a nested
    # percent escape can change privacy semantics and fails closed.
    active: set[str] = set()
    operations = 0
    allowed = min(publication_budget[0], file_budget[0])

    def finish(result: bool) -> bool:
        publication_budget[0] -= operations
        file_budget[0] -= operations
        return result

    def charge(amount: int) -> None:
        nonlocal operations
        operations += max(1, amount)
        if operations > allowed:
            raise _PublicationMatchLimit(
                "Publication encoded-fragment operation limit exceeded"
            )

    for item in tokens:
        if not isinstance(item, tuple) or not item:
            continue
        raw_token = item[0]
        if raw_token == "\0" and len(item) >= 2 and not item[1]:
            active.clear()
            continue
        selectable = len(item) >= 4 and item[3] is True
        if not selectable:
            active.clear()
            continue
        charge(len(raw_token) + len(active))
        additions = set(active)
        for prefix in active:
            for offset in range(len(raw_token)):
                charge(1 + len(prefix) + min(len(raw_token) - offset, 40))
                probe = raw_token[offset:offset + 40]
                completion_length = _encoded_prefix_completion_length(
                    prefix, probe,
                )
                if completion_length is not None:
                    encoded = prefix + probe[:completion_length]
                    charge(len(encoded))
                    if any(
                        variant != encoded and any(
                            character == "%" or character.isspace()
                            or character != "\ufffd" and character.isprintable()
                            for character in variant
                        )
                        for variant in _decoded_variants(encoded)
                    ):
                        return finish(True)
                elif len(raw_token) - offset <= 40:
                    continuation = raw_token[offset:]
                    combined = prefix + continuation
                    additions.update(_incomplete_encoded_suffixes(combined))
        additions.update(
            _incomplete_encoded_suffixes(raw_token)
        )
        # A selectable hidden token may contribute a prefix ending before its
        # physical end. Retain only bounded carriers that actually end in a
        # live escape prefix; ordinary interior text creates no state.
        for endpoint in range(1, len(raw_token)):
            charge(1 + min(endpoint, 40))
            suffixes = _incomplete_encoded_suffixes(
                raw_token[max(0, endpoint - 40):endpoint]
            )
            if not suffixes:
                continue
            additions.update(suffixes)
        if len(additions) > _MAX_ENCODED_STATES:
            raise _PublicationMatchLimit(
                "Publication encoded-fragment state limit exceeded"
            )
        active = additions
    return finish(False)


@lru_cache(maxsize=8192)
def _private_key(key: str) -> bool:
    key = str(key)
    lowered_input = key.lower()
    if not any(marker in lowered_input for marker in (
        "auth", "bearer", "cookie", "credential", "gpu", "host", "job",
        "key", "pass", "private", "scheduler", "secret", "session", "ssh",
        "token", "user", "checkpoint", "authority", "pwd",
    )):
        return False
    normalized = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", str(key))
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", normalized).strip("_")
    lowered = normalized.lower()
    collapsed = lowered.replace("_", "")
    return (
        lowered in {
            "authorization", "auth", "token", "cookie", "session", "session_id",
            "private_key", "ssh_key", "pwd", "passphrase", "dbpass",
        }
        or collapsed in {
            "accesstoken", "apikey", "authoritypath", "authtoken",
            "checkpointpath", "gpuuuid", "jobid", "privatekey",
            "schedulerid", "sessionid", "sshkey", "secretkey", "authkey",
            "signingkey", "encryptionkey", "accesskeyid",
        }
        or collapsed.endswith((
            "token", "privatekey", "sshkey", "cookie", "password",
            "passwd", "secret", "secrets", "credential", "credentials",
            "username", "hostname", "authorization", "authheader", "auth",
            "bearer", "session", "sessionid", "host", "accesskey",
            "accesskeyid",
            "privatekeyid", "sshkeypath",
            "secretkey", "authkey", "signingkey", "encryptionkey",
            "passphrase", "dbpass",
        ))
        or lowered.endswith(("_token", "_private_key", "_ssh_key", "_cookie"))
        or bool(_SENSITIVE_KEY.search(normalized))
    )


def _private_value(value) -> bool:
    return isinstance(value, (str, int, float)) and not isinstance(value, bool) and str(value).strip().lower() not in _EMPTY_PRIVATE_VALUES


def _private_url(value: str) -> bool:
    """Match complete public authorities; never hide URL paths or query secrets."""
    try:
        parsed = urlsplit(value)
        if parsed.hostname not in PUBLIC_HOSTS or parsed.username or parsed.password or parsed.port:
            return True
        query = parse_qsl(parsed.query, keep_blank_values=True)
        if any(_private_key(key) or key.lower() in {"key", "signature"} for key, _ in query):
            return True
        return any(
            _PRIVATE_URL_PATH.search(part)
            for part in (parsed.path, parsed.fragment, *(item for _, item in query))
        )
    except ValueError:
        return True


def _active_attribute(value: str) -> bool:
    """Reject executable URL schemes anywhere in a decoded HTML attribute."""
    return any(
        _ACTIVE_SCHEME.search(re.sub(r"[\x00-\x20]+", "", variant))
        for variant in _decoded_variants(value)
    )


def _remote_runtime_reference(value: str) -> bool:
    """Recognize an encoded external URL used by a runtime-bearing element."""
    for variant in _runtime_url_variants(value):
        parsed = urlsplit(variant.strip())
        if parsed.scheme or parsed.netloc or variant.strip().startswith("//"):
            return True
    return False


def _runtime_url_variants(value: str) -> tuple[str, ...]:
    """Decode URL spellings and apply browser backslash path normalization."""
    return tuple(dict.fromkeys(
        variant.replace("\\", "/") for variant in _decoded_variants(value)
    ))


def _safe_relative_publication_reference(value: str) -> bool:
    """Recognize a local navigation value that cannot encode a private path."""
    for variant in _runtime_url_variants(value):
        stripped = variant.strip()
        try:
            parsed = urlsplit(stripped)
        except ValueError:
            return False
        if parsed.scheme or parsed.netloc:
            if _private_url(stripped):
                return False
            continue
        if (
            stripped.startswith(("/", "\\"))
            or _PRIVATE_URL_PATH.search(parsed.path)
            or any(PATTERNS[index].search(stripped) for index in (11, 12, 14, 15))
        ):
            return False
    return True


def _safe_html_metadata_value(attribute: str, value: str) -> bool:
    """Recognize closed standard vocabularies that cannot carry path data."""
    lowered = " ".join(_decoded(value).lower().split())
    if attribute == "rel":
        return bool(lowered) and set(lowered.split()) <= {
            "alternate", "author", "bookmark", "canonical", "external", "help",
            "icon", "license", "manifest", "me", "next", "nofollow", "noopener",
            "noreferrer", "opener", "prev", "search", "stylesheet", "tag",
        }
    if attribute == "type":
        return lowered in {
            "button", "checkbox", "color", "date", "datetime-local", "email",
            "file", "hidden", "image", "month", "number", "password", "radio",
            "range", "reset", "search", "submit", "tel", "text", "time", "url",
            "week", "text/css", "text/javascript", "application/javascript",
            "application/json", "application/ld+json",
        }
    return False


def _hidden_inline_style(value: str) -> bool:
    """Recognize inline CSS that unambiguously suppresses element rendering."""
    return any(
        re.search(
            r"(?:^|;)\s*(?:display\s*:\s*none|visibility\s*:\s*hidden)"
            r"(?:\s*!important)?\s*(?:;|$)",
            variant, re.I,
        ) is not None
        for variant in _decoded_variants(value)
    )


def _unsafe_css(text: str) -> bool:
    """Reject CSS capable of loading active data or synthesizing hidden text."""
    variants = _decoded_variants(text)
    # CSS escapes can first decode to backslashes; WHATWG URL parsing then
    # normalizes those as slashes for special schemes. Inspect the raw and all
    # decoded stages conservatively instead of trusting only the final stage.
    if "\\" in text and any(
        re.search(r"(?:url\s*\(|@import\b)", variant, re.I)
        for variant in variants
    ):
        return True
    for variant in variants:
        normalized = re.sub(r"/\*.*?\*/", "", variant, flags=re.S)
        compact = re.sub(r"[\x00-\x20]+", "", normalized)
        if (
            _ACTIVE_SCHEME.search(compact) or _REMOTE_CSS_URL.search(compact)
            or _CSS_GENERATED_TEXT.search(normalized)
        ):
            return True
        for match in _CSS_CONTENT.finditer(normalized):
            value = re.sub(r"\s*!important\s*$", "", match.group(1), flags=re.I).strip()
            if value not in _SAFE_CSS_CONTENT_VALUES:
                return True
    return False


def _safe_type_node(node: ast.AST) -> bool:
    """Recognize a deliberately small, value-free Python annotation grammar."""
    if isinstance(node, ast.Name):
        return node.id in _TYPE_ANNOTATION_VALUES
    if isinstance(node, ast.Constant):
        return node.value is None or node.value is Ellipsis
    if isinstance(node, ast.Attribute):
        parts = []
        current = node
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        if not isinstance(current, ast.Name):
            return False
        qualified = ".".join([current.id, *reversed(parts)])
        return qualified in _TYPE_ANNOTATION_VALUES
    if isinstance(node, ast.Subscript):
        return _safe_type_node(node.value) and _safe_type_node(node.slice)
    if isinstance(node, ast.Tuple):
        return bool(node.elts) and all(_safe_type_node(item) for item in node.elts)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        return _safe_type_node(node.left) and _safe_type_node(node.right)
    return False


def _safe_type_assignment(item: str, match: re.Match) -> bool:
    """Confirm that a colon value is a complete annotation, never a prefix."""
    if (
        match.group("delimiter") != ":" or match.group("key_quote")
        or match.group("value").startswith(("\"", "'"))
    ):
        return False
    line_end = item.find("\n", match.start("key"))
    if line_end < 0:
        line_end = len(item)
    if line_end - match.start("key") > _MAX_TYPE_ANNOTATION_BYTES:
        # Parsing one suffix per apparent parameter would otherwise be
        # quadratic. A real Python signature above this bound is not a safe
        # publication exception, so classify it as an ordinary private field.
        return False
    fragment = item[match.start("key"):line_end].strip()
    if fragment.endswith("¶"):
        fragment = fragment[:-1].rstrip()
    fragment = fragment.replace("→", "->")
    if "#" in fragment or ";" in fragment or ("->" in fragment and ")" not in fragment):
        return False
    try:
        if ")" in fragment:
            parsed = ast.parse("def projected(" + fragment + ":\n    pass\n")
        else:
            parsed = ast.parse("def projected(" + fragment + "):\n    pass\n")
    except (SyntaxError, ValueError):
        return False
    function = parsed.body[0]
    arguments = [
        *getattr(function.args, "posonlyargs", ()), *function.args.args,
        *function.args.kwonlyargs,
    ]
    if function.args.vararg is not None:
        arguments.append(function.args.vararg)
    if function.args.kwarg is not None:
        arguments.append(function.args.kwarg)
    if (
        not arguments or arguments[0].arg != match.group("key")
        or not _safe_type_node(arguments[0].annotation)
        or any(argument.annotation is None for argument in arguments)
    ):
        return False
    defaults = [*function.args.defaults, *(
        value for value in function.args.kw_defaults if value is not None
    )]
    if any(
        not isinstance(value, ast.Constant)
        or value.value not in {None, True, False, Ellipsis}
        for value in defaults
    ):
        return False
    if ")" in fragment:
        # The synthetic function parse above consumed the complete signature
        # line, including every later parameter and any return annotation.
        return True
    remainder = item[line_end + 1:] if line_end < len(item) else ""
    if not remainder.strip():
        return True
    provenance = next((line.strip() for line in remainder.splitlines() if line.strip()), "")
    return re.fullmatch(
        r'Source:\s+(?:"|``)?src/[A-Za-z0-9_./-]+:\d+(?:"|``)?\.?',
        provenance,
    ) is not None


def _rst_option_reference(item: str, match: re.Match) -> bool:
    """Distinguish a following ``:type:`` directive option from a value."""
    between = item[match.end("key"):match.start("value")]
    return "\n" in between and re.fullmatch(
        r"[A-Za-z][A-Za-z0-9_-]*:", match.group("value"),
    ) is not None


def violations(text: str, *, field_assignments: bool = True,
               html_markup: bool = False,
               semantic_projection: bool = False) -> list[str]:
    """Return rule identifiers, never the private matched values."""
    result = []
    for value in _decoded_variants(text):
        # Closing markup in a generated RST raw-HTML block is syntax, not a path.
        value = re.sub(r"</[A-Za-z][A-Za-z0-9:_-]*\s*>", "", value)
        lowered = value.lower()
        selected_patterns = _SEMANTIC_PATTERNS if semantic_projection else PATTERNS
        if _PRIVATE_KEY_HEADER.search(value):
            # A header is never publishable by itself. Rejecting immediately is
            # both conservative and avoids quadratic backtracking over repeated
            # unterminated PEM headers in the full block expression.
            result.append("private-pattern-0")
        result.extend(
            "private-pattern-" + str(index)
            for index, pattern in enumerate(selected_patterns)
            if index != 0
            if any(marker in lowered for marker in _PATTERN_MARKERS[index])
            and pattern.search(value)
        )
        if field_assignments:
            assignment_source = re.sub(r"<[^>]*>", "", value) if html_markup else value
            assignment_values = [assignment_source]
            without_comments = re.sub(r"/\*.*?\*/|//[^\r\n]*", "", assignment_source, flags=re.S)
            if without_comments != assignment_source:
                assignment_values.append(without_comments)
            assignment_values.extend(
                re.sub(r"\[\s*([\"'])([A-Za-z_][A-Za-z0-9_-]*)\1\s*\]", r" \2 ", item)
                for item in tuple(assignment_values)
            )
            assignment_values = list(dict.fromkeys(assignment_values))
            if any(
                _private_key(match.group("key"))
                and _private_value(match.group("value").strip("\"'"))
                and not _safe_type_assignment(item, match)
                and not _rst_option_reference(item, match)
                for item in assignment_values for match in _FIELD_ASSIGNMENT.finditer(item)
            ):
                result.append("private-field-assignment")
        if any(_private_url(match.group()) for match in _URL.finditer(value)):
            result.append("private-url")
    return sorted(set(result))


def redact(text: str) -> str:
    """Redact private operational values from AST documentation text."""
    result = _decoded(text)
    if _PRIVATE_KEY_HEADER.search(result):
        return REDACTED
    result = _URL.sub(lambda match: REDACTED if _private_url(match.group()) else match.group(), result)
    for pattern in PATTERNS[1:]:
        result = pattern.sub(REDACTED, result)
    return result


def safe_signature(signature: str) -> str:
    """Keep parameter structure while withholding every default expression."""
    if "=" in signature:
        try:
            name = re.match(r"[A-Za-z_][A-Za-z0-9_.]*", signature)
            if name is None:
                raise ValueError("Invalid signature name")
            qualified = name.group()
            tree = ast.parse("def callable" + signature[name.end():] + ":\n    pass\n")
            function = tree.body[0]
            function.args.defaults = [ast.Constant(Ellipsis) for _ in function.args.defaults]
            function.args.kw_defaults = [ast.Constant(Ellipsis) if value is not None else None for value in function.args.kw_defaults]
            signature = qualified + "(" + ast.unparse(function.args) + ")"
            if function.returns is not None:
                signature += " -> " + ast.unparse(function.returns)
        except (SyntaxError, ValueError):
            # A malformed signature must not leak a value through a best-effort
            # regex. Its callable name remains enough to locate provenance.
            name = re.match(r"[A-Za-z_][A-Za-z0-9_.]*", signature)
            signature = (name.group() if name else "callable") + "(...)"
    return redact(signature)


def project_strings(value):
    """Recursively sanitize a previously allowlisted projection, not raw input."""
    if isinstance(value, str):
        return redact(value)
    if isinstance(value, list):
        return [project_strings(item) for item in value]
    if isinstance(value, dict):
        return {key: project_strings(item) for key, item in value.items()}
    return value


def sensitive_literals(root: Path) -> set[str]:
    """Extract deny-list values locally; these values are never published."""
    from wiki_repository import repository_files
    found = set()
    def walk(value, private=False):
        if isinstance(value, dict):
            # A JSON Schema descriptor describes a field; its type/description
            # values are not recorded credentials. Keep real nested value
            # records subject to the ordinary recursive deny-list extraction.
            if private and isinstance(value.get("type"), str) and value["type"] in {"string", "integer", "number", "boolean", "object", "array", "null"} and set(value).issubset({"type", "description", "title", "format", "pattern", "enum", "const", "default", "examples", "items", "properties", "required", "additionalProperties", "minimum", "maximum", "minLength", "maxLength"}):
                for key in (
                    "const", "default", "enum", "examples", "properties",
                    "items", "additionalProperties",
                ):
                    if key in value:
                        walk(value[key], True)
                return
            for key, item in value.items():
                walk(item, private or _private_key(key))
        elif isinstance(value, list):
            for item in value:
                walk(item, private)
        elif private and isinstance(value, bytes):
            try:
                token = _decoded(value.decode("utf-8")).strip()
            except UnicodeError as error:
                raise ValueError("Unreadable private structured value") from error
            if len(token) >= 4:
                found.add(token)
        elif private and _private_value(value):
            token = _decoded(str(value)).strip()
            if len(token) >= 4:
                found.add(token)
    for path in repository_files(root):
        suffix = path.suffix.lower()
        if suffix not in {".json", ".yaml", ".yml"}:
            continue
        if path.stat().st_size > _MAX_SOURCE_TEXT_BYTES:
            raise ValueError("Structured privacy source exceeds scan limit")
        try:
            if suffix == ".json":
                walk(_strict_json(path.read_text()))
            else:
                walk(_strict_yaml(path.read_text()))
        except _DuplicateStructuredKey:
            raise ValueError("Duplicate key in structured privacy source") from None
        except (ValueError, UnicodeError, OSError, _UnsafeEncoding):
            raise ValueError("Unreadable structured privacy source") from None
        except Exception as error:
            # Parser exceptions can embed source content and paths. Fail closed
            # with a constant diagnostic rather than skipping a deny-list input.
            if error.__class__.__module__.startswith("yaml."):
                raise ValueError("Unreadable structured privacy source") from None
            raise
    return found


def _raw_start_tag_parts(
    value: str,
) -> tuple[list[str], list[str], list[tuple[str, bool, bool, bool]]]:
    """Lex case-preserving names and values in their raw lexical order."""
    names, tokens, ordered = [], [], []
    index = value.find("<") + 1
    length = len(value)
    while index < length and value[index].isspace():
        index += 1
    if index < length and value[index] == "/":
        index += 1
    start = index
    while index < length and not value[index].isspace() and value[index] not in "/>":
        index += 1
    tag_name = value[start:index].lower()
    if index > start:
        names.append(value[start:index])
        tokens.append(value[start:index])
        ordered.append((
            value[start:index], True, True,
            tag_name not in _STANDARD_HTML_ELEMENTS,
        ))
    while index < length:
        while index < length and value[index].isspace():
            index += 1
        if index >= length or value[index] == ">":
            break
        if value[index] == "/":
            if re.fullmatch(r"/\s*>", value[index:]) is not None:
                break
            index += 1
            continue
        start = index
        while index < length and not value[index].isspace() and value[index] not in "=/>":
            index += 1
        attribute_name = value[start:index].lower()
        if index > start:
            names.append(value[start:index])
            tokens.append(value[start:index])
            ordered.append((
                value[start:index], True, True,
                attribute_name not in _STANDARD_HTML_ATTRIBUTES
                and not attribute_name.startswith(("aria-", "data-", "on")),
            ))
        while index < length and value[index].isspace():
            index += 1
        if index >= length or value[index] != "=":
            continue
        index += 1
        # Preserve the browser-visible assignment relationship for the typed
        # tag scanner.  The flat token list remains names/values-only for
        # compatibility checks; ordered security reconstruction includes it.
        ordered.append(("=", False, True, True))
        while index < length and value[index].isspace():
            index += 1
        if index < length and value[index] in "\"'":
            quote = value[index]
            index += 1
            start = index
            while index < length and value[index] != quote:
                index += 1
            if index > start:
                raw_value = value[start:index]
                tokens.append(raw_value)
                path_startable = not (
                    (
                        (tag_name, attribute_name) in _RUNTIME_URL_ATTRIBUTES
                        or attribute_name in {"href", "xlink:href", "action"}
                    )
                    and _safe_relative_publication_reference(raw_value)
                    or _safe_html_metadata_value(attribute_name, raw_value)
                )
                ordered.append((raw_value, False, True, path_startable))
            index += index < length
        else:
            # In HTML an unquoted value may contain ``/``. It is a self-close
            # marker only after intervening whitespace, which the outer loop
            # handles before inspecting the next attribute/token.
            start = index
            while index < length and not value[index].isspace() and value[index] != ">":
                index += 1
            if index > start:
                raw_value = value[start:index]
                tokens.append(raw_value)
                path_startable = not (
                    (
                        (tag_name, attribute_name) in _RUNTIME_URL_ATTRIBUTES
                        or attribute_name in {"href", "xlink:href", "action"}
                    )
                    and _safe_relative_publication_reference(raw_value)
                    or _safe_html_metadata_value(attribute_name, raw_value)
                )
                ordered.append((raw_value, False, True, path_startable))
    return names, tokens, ordered


def _raw_start_tag_names(value: str) -> list[str]:
    """Return only the case-preserving name channel for compatibility checks."""
    return _raw_start_tag_parts(value)[0]


def _semantic_tokens(value: str) -> list[str]:
    """Split one parser channel while retaining canonical word separators."""
    return [" " if part.isspace() else part for part in re.findall(r"\S+|\s+", value)]


def _raw_markup_channels(
    value: str,
) -> tuple[
    list[str], list[list[tuple[str, bool, bool]]],
    list[
        tuple[str, bool, bool] | tuple[str, bool, bool, bool]
        | tuple[str, bool, bool, bool, bool]
    ], bool,
]:
    """Collect raw channels and report whether every markup construct closed."""
    names, token_groups, ordered = [], [], []
    hidden_stack: list[tuple[str, bool]] = []

    def append_data(data: str) -> None:
        if not data:
            return
        if (
            hidden_stack and hidden_stack[-1][0] == "script"
            and data.strip() in _SAFE_INLINE_SCRIPTS
        ):
            ordered.append(("\0", False, False))
            return
        if hidden_stack and hidden_stack[-1][1] and not data.isspace():
            ordered.append((data, False, True, True))
        else:
            ordered.append((data, False, False))

    complete = True
    index = 0
    while index < len(value):
        start = value.find("<", index)
        if start < 0:
            append_data(value[index:])
            break
        append_data(value[index:start])
        if start + 1 >= len(value):
            # HTMLParser exposes a lone less-than sign as ordinary data.
            append_data(value[start:])
            break
        if value.startswith("<!--", start):
            end = value.find("-->", start + 4)
            if end < 0:
                complete = False
            content_end = len(value) if end < 0 else end
            if value[start + 4:content_end]:
                ordered.append((
                    value[start + 4:content_end], False, True, True,
                ))
            index = len(value) if end < 0 else end + 3
            continue
        if value[start + 1] in "!?":
            quote = None
            end = start + 2
            while end < len(value):
                character = value[end]
                if quote is None and character in "\"'":
                    quote = character
                elif quote is not None and character == quote:
                    quote = None
                elif quote is None and character == ">":
                    break
                end += 1
            if end >= len(value):
                complete = False
                end = -1
            content_end = len(value) if end < 0 else end
            directive = value[start + 2:content_end]
            match = re.fullmatch(r"(\s*\S+\s*)(.*)", directive, re.S)
            if match is None:
                if directive:
                    ordered.append((directive, True, True, True))
            else:
                ordered.append((match.group(1), True, True, True))
                if match.group(2):
                    ordered.append((match.group(2), False, True, True))
            index = len(value) if end < 0 else end + 1
            continue
        if re.match(r"</?[A-Za-z]", value[start:]) is None:
            # Preserve malformed-but-textual less-than sequences exactly as the
            # standard parser does; they are not incomplete tag constructs.
            append_data("<")
            index = start + 1
            continue
        quote = None
        end = start + 1
        while end < len(value):
            char = value[end]
            if quote is None and char in "\"'":
                quote = char
            elif quote is not None and char == quote:
                quote = None
            elif quote is None and char == ">":
                break
            end += 1
        if end >= len(value):
            complete = False
            raw_tag = value[start:]
        else:
            raw_tag = value[start:end + 1]
        tag_names, tag_tokens, tag_ordered = _raw_start_tag_parts(raw_tag)
        names.extend(tag_names)
        if tag_tokens:
            token_groups.append([item[:3] for item in tag_ordered])
            # Every start-tag token is absent from visible document text. Keep
            # its name/value fragments selectable for reconstruction while
            # allowing unrelated markup tokens to be skipped. The fourth flag
            # is intentionally distinct from substring/noisy-carrier matching.
            ordered.extend(
                (token, structural, allow_noise, True, path_startable)
                for token, structural, allow_noise, path_startable in tag_ordered
            )
        syntactic_self_close = re.search(r"/\s*>$", raw_tag) is not None
        if tag_names and syntactic_self_close:
            # HTMLParser represents a syntactic self-close as start + end.
            names.append(tag_names[0])
        if tag_names:
            tag_name = tag_names[0].lower()
            if syntactic_self_close and tag_name not in _VOID_HTML_ELEMENTS:
                complete = False
            if re.match(r"<\s*/", raw_tag):
                for stack_index in range(len(hidden_stack) - 1, -1, -1):
                    if hidden_stack[stack_index][0] == tag_name:
                        del hidden_stack[stack_index:]
                        break
            elif (
                tag_name not in _VOID_HTML_ELEMENTS
                and not syntactic_self_close
            ):
                attribute_names = {name.lower() for name in tag_names[1:]}
                style_values = [
                    tag_ordered[token_index + 2][0]
                    for token_index, token in enumerate(tag_ordered[:-2])
                    if token[1] and token[0].lower() == "style"
                    and tag_ordered[token_index + 1][0] == "="
                ]
                hidden_stack.append((
                    tag_name,
                    bool(hidden_stack and hidden_stack[-1][1])
                    or tag_name in _NONVISIBLE_HTML_ELEMENTS
                    or tag_name == "details" and "open" not in attribute_names
                    or "hidden" in attribute_names
                    or any(_hidden_inline_style(item) for item in style_values),
                ))
        index = len(value) if end >= len(value) else end + 1
    return names, token_groups, ordered, complete


def _raw_markup_names(value: str) -> list[str]:
    """Collect names from every quote-aware raw start and end tag."""
    return _raw_markup_channels(value)[0]


def _css_comment_tokens(value: str) -> list[tuple[str, bool, bool, bool]]:
    """Lex actual CSS comment bodies without mistaking quoted text for comments."""
    tokens = []

    def append_comment(body: str) -> None:
        """Exclude complete allowlisted public URLs from path reconstruction."""
        cursor = 0
        for match in _URL.finditer(body):
            if _private_url(match.group(0)):
                continue
            if match.start() > cursor:
                tokens.append((body[cursor:match.start()], False, True, True))
            tokens.append(("\0", False, False))
            cursor = match.end()
        if cursor < len(body):
            tokens.append((body[cursor:], False, True, True))

    quote = None
    index = 0
    while index < len(value):
        character = value[index]
        if quote is not None:
            if character == "\\":
                index += 2
                continue
            if character == quote:
                quote = None
            index += 1
            continue
        if character in "\"'":
            quote = character
            index += 1
            continue
        if value.startswith("/*", index):
            end = value.find("*/", index + 2)
            if end < 0:
                raise ValueError("Unterminated CSS comment")
            if end > index + 2:
                append_comment(value[index + 2:end])
            index = end + 2
            continue
        index += 1
    return tokens


def _structured_ordered_tokens(value) -> list[tuple]:
    """Expose direct scalar siblings while keeping container boundaries distinct."""
    streams: list[list[tuple[str, bool, bool, bool]]] = []

    def walk(item) -> None:
        nested = []
        if isinstance(item, dict):
            for key, child in item.items():
                if isinstance(child, (dict, list)):
                    nested.append(child)
                elif isinstance(child, (str, int, float)) and not isinstance(child, bool):
                    # A mapping exposes a key/value relationship, not one
                    # concatenated stream spanning every unrelated entry.
                    # Keeping each pair distinct avoids manufacturing private
                    # values from large generated manifests while retaining
                    # selectable sibling reconstruction for actual arrays.
                    streams.append([
                        (str(key), False, True, True),
                        (str(child), False, True, True),
                    ])
        elif isinstance(item, list):
            current = []
            for child in item:
                if isinstance(child, (dict, list)):
                    nested.append(child)
                elif isinstance(child, (str, int, float)) and not isinstance(child, bool):
                    current.append((str(child), False, True, True))
            if current:
                streams.append(current)
        for child in nested:
            walk(child)

    walk(value)
    result = []
    for stream in streams:
        if result:
            result.append(("\0", False, False))
        result.extend(stream)
    return result


def _assignment_ordered_regions(tokens: list[tuple]) -> list[tuple]:
    """Retain only reset-delimited channels containing an assignment delimiter."""
    result = []
    region = []
    for item in [*tokens, ("\0", False, False)]:
        if len(item) == 3 and item[0] == "\0":
            if region and any(
                ":" in variant or "=" in variant
                for token in region
                for variant in _decoded_variants(token[0])
            ):
                if result:
                    result.append(("\0", False, False))
                result.extend(region)
            region = []
        else:
            region.append(item)
    return result


def _field_state_suffix(value: str) -> str | None:
    """Retain a bounded identifier suffix that can become a private key."""
    identifier = re.search(r"[A-Za-z_][A-Za-z0-9_-]*$", value)
    return identifier.group(0)[-128:] if identifier is not None else None


def _ordered_tag_violation(
    tokens: list[tuple[str, bool, bool]],
    publication_budget: list[int],
    file_budget: list[int],
) -> bool:
    """Scan one raw tag with structural include-or-skip NFA semantics."""
    def standard_accesskey(value: str) -> bool:
        return bool(value.split()) and all(
            len(item) == 1 for item in value.split()
        )

    # ``accesskey`` is a standard HTML keyboard-shortcut attribute, while the
    # same key name in JSON/YAML remains credential-sensitive. Remove only a
    # lexically complete, valid shortcut triplet from this tag-local scanner.
    filtered = []
    token_index = 0
    while token_index < len(tokens):
        token, structural, allow_noise = tokens[token_index]
        if (
            token_index > 0 and structural and token.lower() == "accesskey"
            and token_index + 2 < len(tokens)
            and tokens[token_index + 1][0] == "="
            and all(
                standard_accesskey(variant)
                for variant in _decoded_variants(tokens[token_index + 2][0])
            )
        ):
            token_index += 3
            continue
        filtered.append((token, structural, allow_noise))
        token_index += 1
    tokens = filtered
    states: set[str] = set()
    operations = 0
    allowed = min(publication_budget[0], file_budget[0])

    def finish(result: bool) -> bool:
        publication_budget[0] -= operations
        file_budget[0] -= operations
        return result

    decoded_tokens = []
    has_split_delimiter = False
    for raw_token, structural, _allow_noise in tokens:
        variants = _decoded_variants(raw_token)
        decoded_tokens.append((variants, structural))
        for variant in variants:
            operations += len(variant) + 1
            if operations > allowed:
                raise _PublicationMatchLimit(
                    "Publication matcher operation limit exceeded"
                )
            if "private-field-assignment" in violations(variant):
                return finish(True)
            if ":" in variant or "=" in variant:
                has_split_delimiter = True
    if not has_split_delimiter:
        return finish(False)

    for variants, _structural in decoded_tokens:
        frozen = tuple(states)
        additions = set(frozen)
        # Each typed name/value token is selectable or skippable. Alternatives
        # all advance from the same frozen set so encoded spellings cannot feed
        # each other within one token. Only tags with an actual delimiter enter
        # this NFA, keeping ordinary generated markup linear.
        for variant in variants:
            for prefix in ("", *frozen):
                candidate = prefix + variant
                operations += len(candidate) + 1
                if operations > allowed:
                    raise _PublicationMatchLimit(
                        "Publication matcher operation limit exceeded"
                    )
                if "private-field-assignment" in violations(candidate):
                    return finish(True)
                assignment_head = re.search(
                    r"([A-Za-z_][A-Za-z0-9_-]*)\s*[:=]", candidate,
                )
                if (
                    assignment_head is not None
                    and _private_key(assignment_head.group(1))
                ):
                    return finish(True)
                retained = _field_state_suffix(candidate)
                if retained:
                    additions.add(retained)
        if len(additions) > _MAX_MATCH_STATES:
            raise _PublicationMatchLimit(
                "Publication matcher state limit exceeded"
            )
        states = additions
    return finish(False)


class _PublishedHTML(HTMLParser):
    """Collect every HTML parser channel and sensitive name/value relationship."""
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self.content = []
        self.compounds = []
        self.names = []
        self.raw_names = []
        self.visible = []
        self.semantic = []
        self.semantic_names_visible = []
        self.semantic_values_visible = []
        self.hidden = []
        self.private_field_value = False
        self.private_path_value = False
        self.duplicate_attribute = False
        self.active_content = False
        self.script_sources = []
        self.form_actions = []
        self._script = None
        self._hidden_stack: list[tuple[str, bool]] = []

    def handle_data(self, data):
        self.parts.append(data)
        self.content.append(data)
        self.semantic.append(data)
        if self._script is not None and data.strip() in _SAFE_INLINE_SCRIPTS:
            self.hidden.append("\0")
        elif (
            self._hidden_stack and self._hidden_stack[-1][1]
            and not data.isspace()
        ):
            self.hidden.append(data)
        else:
            self.visible.append(data)
            self.semantic_names_visible.append(data)
            self.semantic_values_visible.append(data)
            self.hidden.append("\0")
        if self._script is not None:
            self._script["body"].append(data)

    def handle_comment(self, data):
        self.parts.append(data)
        self.content.append(data)
        self.semantic.append(data)
        self.hidden.append(data.strip())

    def handle_starttag(self, tag, attrs):
        tag = tag.lower()
        raw_tag = self.get_starttag_text()
        if raw_tag:
            _names, _tokens, ordered = _raw_start_tag_parts(raw_tag)
            self.semantic.extend(
                token for token_index, (
                    token, _structural, _allow_noise, _path_startable,
                )
                in enumerate(ordered)
                if token_index > 0 and token != "="
            )
            self.semantic_names_visible.extend(
                token for token_index, (
                    token, structural, _allow_noise, _path_startable,
                )
                in enumerate(ordered)
                if token_index > 0 and structural
            )
            self.semantic_values_visible.extend(
                token for token_index, (
                    token, structural, _allow_noise, _path_startable,
                )
                in enumerate(ordered)
                if token_index > 0 and not structural and token != "="
            )
        self.parts.append(tag)
        self.names.append(tag)
        attribute_names = [name.lower() for name, _ in attrs]
        if len(attribute_names) != len(set(attribute_names)):
            self.duplicate_attribute = True
        attributes = {
            name.lower(): value for name, value in attrs if value is not None
        }
        self.compounds.append(tag + "".join(attribute_names))
        self.compounds.append("".join(value for _, value in attrs if value))
        for name, value in attrs:
            self.parts.append(name)
            self.names.append(name)
            if value:
                self.parts.append(value)
                self.content.append(value)
            safe_accesskey = (
                name.lower() == "accesskey" and bool(str(value).split())
                and all(len(item) == 1 for item in str(value).split())
            )
            if _private_key(name) and _private_value(value) and not safe_accesskey:
                self.private_field_value = True
            if value and any(
                _PRIVATE_URL_PATH.search(urlsplit(variant).path)
                for variant in _runtime_url_variants(value)
            ):
                self.private_path_value = True
        if tag in {
            "iframe", "object", "embed", "applet", "style", "base", "svg",
        }:
            self.active_content = True
        for name, value in attrs:
            lowered_name = name.lower()
            if (
                lowered_name == "attributionsrc"
                or value and (
                    lowered_name.startswith("on")
                or lowered_name in {
                    "formaction", "formmethod", "ping",
                }
                or tag in {"image", "use"}
                and lowered_name in {"href", "xlink:href"}
                )
            ):
                self.active_content = True
            if value and (
                lowered_name in {"srcset", "imagesrcset"}
                or _active_attribute(value)
                or lowered_name == "style" and (
                    "/*" in value or "*/" in value or _unsafe_css(value)
                )
                or (tag, lowered_name) in _RUNTIME_URL_ATTRIBUTES
                and _remote_runtime_reference(value)
            ):
                self.active_content = True
        if tag == "input" and str(attributes.get("type", "")).lower() == "image":
            self.active_content = True
        if tag == "form":
            action = str(attributes.get("action", ""))
            if (
                str(attributes.get("method", "get")).lower() != "get"
                or _remote_runtime_reference(action)
                or any(
                    urlsplit(variant).path.startswith("/")
                    for variant in _runtime_url_variants(action)
                )
                or _active_attribute(action)
            ):
                self.active_content = True
            else:
                self.form_actions.append(action)
        if tag == "script":
            if self._script is not None:
                self.active_content = True
            source = attributes.get("src")
            if source:
                self.script_sources.append(source)
            self._script = {"src": source, "body": []}
        declared = next(
            (attributes[name] for name in ("name", "property", "itemprop", "http-equiv")
             if name in attributes and _private_key(attributes[name])),
            None,
        )
        if declared is not None and any(
            _private_value(attributes.get(name)) for name in ("content", "value")
        ):
            self.private_field_value = True
        if (
            tag.lower() == "input"
            and str(attributes.get("type", "")).lower() == "password"
            and _private_value(attributes.get("value"))
        ):
            self.private_field_value = True
        if tag == "meta" and "http-equiv" in attributes:
            self.active_content = True
        if tag not in _VOID_HTML_ELEMENTS:
            self._hidden_stack.append((
                tag,
                bool(self._hidden_stack and self._hidden_stack[-1][1])
                or tag in _NONVISIBLE_HTML_ELEMENTS
                or tag == "details" and "open" not in attribute_names
                or "hidden" in attributes
                or _hidden_inline_style(str(attributes.get("style", ""))),
            ))

    def handle_endtag(self, tag):
        tag = tag.lower()
        self.parts.append(tag)
        self.names.append(tag)
        if tag == "script":
            if self._script is None:
                self.active_content = True
                return
            body = "".join(self._script["body"]).strip()
            source = self._script["src"]
            if (source and body) or (not source and body not in _SAFE_INLINE_SCRIPTS):
                self.active_content = True
            self._script = None
        for stack_index in range(len(self._hidden_stack) - 1, -1, -1):
            if self._hidden_stack[stack_index][0] == tag:
                del self._hidden_stack[stack_index:]
                break

    def handle_decl(self, declaration):
        self.parts.append(declaration)
        self.content.append(declaration)
        self.semantic.append(declaration)
        self.hidden.append(declaration.strip())

    def handle_pi(self, data):
        self.parts.append(data)
        self.content.append(data)
        self.semantic.append(data)
        self.hidden.append(data.removesuffix("?").strip())

    def unknown_decl(self, data):
        self.parts.append(data)
        self.content.append(data)
        self.semantic.append(data)
        self.hidden.append(data.strip())

    def close(self):
        super().close()
        if self._script is not None:
            self.active_content = True


def _strings(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield str(key)
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)
    elif isinstance(value, (str, int, float)):
        yield str(value)


def _keys(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield str(key)
            yield from _keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from _keys(item)


def _values(value):
    if isinstance(value, dict):
        for item in value.values():
            yield from _values(item)
    elif isinstance(value, list):
        for item in value:
            yield from _values(item)
    elif isinstance(value, (str, int, float)):
        yield str(value)


def _strict_json(value: str):
    """Decode JSON while rejecting keys whose earlier bytes would be discarded."""
    def unique_object(pairs):
        result = {}
        for key, item in pairs:
            if key in result:
                raise _DuplicateStructuredKey("Duplicate JSON key")
            result[key] = item
        return result

    return json.loads(value, object_pairs_hook=unique_object)


def _strict_yaml(value: str):
    """Decode safe YAML while rejecting mappings that discard earlier values."""
    import yaml

    class UniqueKeyLoader(yaml.SafeLoader):
        pass

    def unique_mapping(loader, node, deep=False):
        loader.flatten_mapping(node)
        result = {}
        for key, item in loader.construct_pairs(node, deep=deep):
            try:
                duplicate = key in result
            except TypeError as error:
                raise ValueError("Unhashable YAML mapping key") from error
            if duplicate:
                raise _DuplicateStructuredKey("Duplicate YAML key")
            result[key] = item
        return result

    UniqueKeyLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
        unique_mapping,
    )
    return yaml.load(value, Loader=UniqueKeyLoader)


def _contains_private_fields(value, private=False) -> bool:
    """Reject actual sensitive field values at arbitrary nesting depth."""
    if isinstance(value, dict):
        return any(_contains_private_fields(item, private or _private_key(key)) for key, item in value.items())
    if isinstance(value, list):
        return any(_contains_private_fields(item, private) for item in value)
    return private and _private_value(value)


def _valid_search_index(value) -> bool:
    """Recognize Sphinx's typed index before treating term keys as index words.

    A search term such as ``username`` maps to document numbers, not an
    operational username field. Unknown keys or arbitrary nested value records
    do not receive this exception. Every index string is still privacy-scanned.
    """
    required = {"docnames", "filenames", "titles", "terms", "titleterms", "objects", "objnames", "objtypes", "envversion"}
    if not isinstance(value, dict) or not required.issubset(value) or set(value) - required - {"alltitles", "indexentries"}:
        return False
    for key in ("docnames", "filenames", "titles"):
        if not isinstance(value[key], list) or not all(isinstance(item, str) for item in value[key]):
            return False
    count = len(value["docnames"])
    if len(value["filenames"]) != count or len(value["titles"]) != count:
        return False

    def doc_index(item):
        return type(item) is int and 0 <= item < count

    for key in ("terms", "titleterms"):
        if not isinstance(value[key], dict):
            return False
        for term, references in value[key].items():
            if not isinstance(term, str) or not (doc_index(references) or isinstance(references, list) and all(doc_index(item) for item in references)):
                return False
    for key in ("objnames", "objtypes"):
        if not isinstance(value[key], dict):
            return False
        for identifier, item in value[key].items():
            if not isinstance(identifier, str) or not identifier.isdigit():
                return False
            if key == "objtypes" and not isinstance(item, str):
                return False
            if key == "objnames" and not (isinstance(item, list) and len(item) == 3 and all(isinstance(part, str) for part in item)):
                return False
    objects_value = value["objects"]
    if not isinstance(objects_value, dict):
        return False
    for prefix, objects in objects_value.items():
        if not isinstance(prefix, str):
            return False
        if isinstance(objects, list):
            # Sphinx 8 compact form: prefix -> records carrying the suffix.
            for item in objects:
                if not (
                    isinstance(item, list) and len(item) == 5
                    and doc_index(item[0]) and type(item[1]) is int
                    and type(item[2]) is int and isinstance(item[3], str)
                    and isinstance(item[4], str)
                ):
                    return False
        elif isinstance(objects, dict):
            # Sphinx 3--7 form: prefix -> suffix -> four-field record.
            for suffix, item in objects.items():
                if not (
                    isinstance(suffix, str) and isinstance(item, list)
                    and len(item) == 4 and doc_index(item[0])
                    and type(item[1]) is int and type(item[2]) is int
                    and isinstance(item[3], str)
                ):
                    return False
        else:
            return False
    for key, width in (("alltitles", 2), ("indexentries", 3)):
        if key not in value:
            continue
        if not isinstance(value[key], dict):
            return False
        for title, entries in value[key].items():
            if not isinstance(title, str) or not isinstance(entries, list):
                return False
            for item in entries:
                if not (isinstance(item, list) and len(item) == width and doc_index(item[0]) and (item[1] is None or isinstance(item[1], str)) and (width == 2 or isinstance(item[2], bool))):
                    return False
    version = value["envversion"]
    return type(version) is int or isinstance(version, dict) and all(isinstance(key, str) and type(item) is int for key, item in version.items()) and not _contains_private_fields(version)


def _search_index_ordered_tokens(value: dict) -> list[tuple]:
    """Expose only related string fields from a validated Sphinx index.

    Numeric document references are typed indexes, never field values. Each
    independent title, term, or filename is reset-delimited, while the strings
    in an object-name or object record remain one selectable logical record.
    """
    streams: list[list[str]] = []

    def add(*parts) -> None:
        strings = [part for part in parts if isinstance(part, str)]
        if strings:
            streams.append(strings)

    # Independent scalar records are already scanned one by one. Only retain
    # schema fields whose adjacent strings describe one object/index record.
    for item in value["objnames"].values():
        add(*item)
    for prefix, objects in value["objects"].items():
        if isinstance(objects, list):
            for item in objects:
                add(prefix, item[3], item[4])
        else:
            for suffix, item in objects.items():
                add(prefix, suffix, item[3])
    result = []
    for stream in streams:
        if result:
            result.append(("\0", False, False))
        result.extend((item, False, True, True) for item in stream)
    return result


def _canonical_body(value: str) -> str:
    # A UTF-8 signature is transport metadata, not part of rendered text.
    return " ".join(_decoded(value).lstrip("\ufeff").split())


def _canonical_variants(value: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(" ".join(item.split()) for item in _decoded_variants(value)))


def _literal_automaton(literals: list[str], publication_budget: list[int]):
    """Compile exact private literals into a bounded Aho-Corasick trie."""
    transitions: list[dict[str, int]] = [{}]
    failures = [0]
    # Validation needs only existence, not the matched private value. A boolean
    # output flag avoids quadratic failure-output sets for nested-prefix rules.
    terminal = [False]
    operations = 0
    allowed = min(_MAX_MATCH_OPERATIONS, publication_budget[0])

    def charge() -> None:
        nonlocal operations
        operations += 1
        if operations > allowed:
            raise _PublicationMatchLimit(
                "Publication matcher operation limit exceeded"
            )

    for literal in literals:
        state = 0
        for character in literal:
            charge()
            next_state = transitions[state].get(character)
            if next_state is None:
                next_state = len(transitions)
                if next_state >= _MAX_MATCH_STATES:
                    raise ValueError("Sensitive literal matcher state limit exceeded")
                transitions[state][character] = next_state
                transitions.append({})
                failures.append(0)
                terminal.append(False)
            state = next_state
        terminal[state] = True
    queue = list(transitions[0].values())
    cursor = 0
    while cursor < len(queue):
        state = queue[cursor]
        cursor += 1
        for character, next_state in transitions[state].items():
            charge()
            fallback = failures[state]
            while fallback and character not in transitions[fallback]:
                charge()
                fallback = failures[fallback]
            failures[next_state] = transitions[fallback].get(character, 0)
            terminal[next_state] = (
                terminal[next_state] or terminal[failures[next_state]]
            )
            queue.append(next_state)
    publication_budget[0] -= operations
    return transitions, failures, terminal


def _literal_automaton_contains(
    matcher,
    value: str,
    publication_budget: list[int],
    file_budget: list[int],
) -> bool:
    """Match one decoded channel in linear, explicitly budgeted time."""
    transitions, failures, terminal = matcher
    allowed = min(publication_budget[0], file_budget[0])
    operations = 0
    state = 0

    def finish(result: bool) -> bool:
        publication_budget[0] -= operations
        file_budget[0] -= operations
        return result

    for character in value:
        operations += 1
        if operations > allowed:
            raise _PublicationMatchLimit(
                "Publication matcher operation limit exceeded"
            )
        while state and character not in transitions[state]:
            operations += 1
            if operations > allowed:
                raise _PublicationMatchLimit(
                    "Publication matcher operation limit exceeded"
                )
            state = failures[state]
        state = transitions[state].get(character, 0)
        if terminal[state]:
            return finish(True)
    return finish(False)


def _raw_body_contains(
    prefixes: dict[str, list[str]],
    prefix_width: int,
    value: str,
    publication_budget: list[int],
    file_budget: list[int],
) -> bool:
    """Bound one prefix-indexed scan for complete source bodies."""
    allowed = min(publication_budget[0], file_budget[0])
    operations = 0

    def charge(amount: int) -> None:
        nonlocal operations
        operations += amount
        if operations > allowed:
            raise _PublicationMatchLimit(
                "Publication matcher operation limit exceeded"
            )

    def finish(result: bool) -> bool:
        publication_budget[0] -= operations
        file_budget[0] -= operations
        return result

    for start in range(max(0, len(value) - prefix_width + 1)):
        charge(1)
        bodies = prefixes.get(value[start:start + prefix_width], ())
        for body in bodies:
            # startswith may compare every remaining body character. Charge its
            # worst-case bounded work, including shared-prefix adversaries.
            charge(min(len(body), len(value) - start))
            if value.startswith(body, start):
                return finish(True)
    return finish(False)


def _canonical_token_variants_uncached(value: str) -> tuple[str, ...]:
    results = []
    for item in _decoded_variants(value):
        normalized = " ".join(item.split())
        if item[:1].isspace():
            normalized = " " + normalized
        if item[-1:].isspace() and not normalized.endswith(" "):
            normalized += " "
        if normalized not in results:
            results.append(normalized)
    return tuple(results)


@lru_cache(maxsize=8192)
def _cached_canonical_token_variants(value: str) -> tuple[str, ...]:
    return _canonical_token_variants_uncached(value)


def _canonical_token_variants(value: str) -> tuple[str, ...]:
    """Canonicalize a token without discarding decoded boundary whitespace."""
    # Generated markup repeats short tag, attribute, class and punctuation
    # carriers heavily. Bound the cache by both entry count and token size so a
    # publication cannot retain a large body of unique page text in memory.
    if len(value) <= 512:
        return _cached_canonical_token_variants(value)
    return _canonical_token_variants_uncached(value)


def _body_prefix_index(bodies: list[str] | tuple[str, ...]):
    """Compile the immutable shared-prefix index once per source corpus."""
    width = 8
    start_work: list[dict[str, list[int]]] = [
        defaultdict(list) for _ in range(width + 1)
    ]
    prefix_children: dict[str, set[str]] = defaultdict(set)
    terminal_prefixes: set[str] = set()
    for index, body in enumerate(bodies):
        for size in range(1, min(width, len(body)) + 1):
            prefix = body[:size]
            start_work[size][prefix].append(index)
            if size < width and size < len(body):
                prefix_children[prefix].add(body[size])
            if size < width and size == len(body):
                terminal_prefixes.add(prefix)
    starts = tuple(
        {prefix: tuple(indices) for prefix, indices in mapping.items()}
        for mapping in start_work
    )
    short_starts = tuple(
        index for index, body in enumerate(bodies) if len(body) < width
    )
    return (
        starts,
        short_starts,
        {prefix: frozenset(children)
         for prefix, children in prefix_children.items()},
        frozenset(terminal_prefixes),
        {},
    )


def _ordered_token_stream_contains(
    bodies: list[str],
    tokens: list[
        str | tuple[str, bool] | tuple[str, bool, bool]
        | tuple[str, bool, bool, bool] | tuple[str, bool, bool, bool, bool]
    ],
    operation_budget: list[int] | None = None,
    body_prefix_index=None,
) -> bool:
    """Match source bodies within a bounded, ordered markup envelope.

    Parser-channel tokens may contain harmless prefixes or suffixes, and markup
    tokens may occur between source fragments. Initial carriers of fewer than
    eight characters stay as shared trie-prefix states; only a confirmed
    eight-character prefix expands to body-specific states. Prefix indexes keep
    the NFA proportional to plausible rendered-source spans instead of comparing
    every body with every token. Established matches receive enough room for
    per-character highlighting markup, with an absolute state bound.
    """
    if not bodies or not tokens:
        return False

    width = 8
    if body_prefix_index is None:
        body_prefix_index = _body_prefix_index(bodies)
    (starts, short_starts, prefix_children, terminal_prefixes,
     noisy_prefix_cache) = body_prefix_index
    active: list[dict[str, set[tuple[int, int]]]] = [
        defaultdict(set) for _ in range(width + 1)
    ]
    terminal: set[tuple[int, int]] = set()
    after_space: set[tuple[int, int]] = set()
    live: dict[tuple[int, int], int] = {}
    state_keys: dict[tuple[int, int], tuple[tuple[int, str], ...]] = {}
    bootstrap: dict[str, int] = {}
    bootstrap_after_space: set[str] = set()
    bootstrap_by_next: dict[str, set[str]] = defaultdict(set)
    operations = 0
    allowed_operations = min(
        _MAX_MATCH_OPERATIONS,
        operation_budget[0] if operation_budget is not None else _MAX_MATCH_OPERATIONS,
    )

    def finish(result: bool) -> bool:
        if operation_budget is not None:
            operation_budget[0] -= operations
        return result

    def charge(amount: int) -> None:
        """Charge a conservative byte-comparison upper bound."""
        nonlocal operations
        operations += max(1, amount)
        if operations > allowed_operations:
            raise ValueError("Publication matcher operation limit exceeded")

    def bounded_contains(haystack: str, needle: str) -> bool:
        # ``str.__contains__`` may inspect the needle at every candidate byte.
        candidates = max(1, len(haystack) - len(needle) + 1)
        charge(candidates * max(1, len(needle)))
        return needle in haystack

    def bounded_prefix(
        value: str, value_start: int, prefix: str, prefix_start: int = 0,
    ) -> bool:
        required = len(prefix) - prefix_start
        available = len(value) - value_start
        charge(min(max(0, available), max(0, required)) + 1)
        if required > available:
            return False
        return value.startswith(prefix[prefix_start:], value_start)

    def add_state(state: tuple[int, int, int]) -> None:
        body_index, position, began = state
        key = (body_index, position)
        previous_began = live.get(key)
        if previous_began is not None and previous_began >= began:
            return
        body = bodies[body_index]
        remaining_length = len(body) - position
        keys = tuple(
            (size, body[position:position + size])
            for size in range(1, min(width, remaining_length) + 1)
        )
        live[key] = began
        if previous_began is None:
            for size, prefix in keys:
                active[size][prefix].add(key)
            if remaining_length < width:
                terminal.add(key)
            if position and body[position - 1] == " ":
                after_space.add(key)
            state_keys[key] = keys
        if len(live) + len(bootstrap) > _MAX_MATCH_STATES:
            raise ValueError("Publication matcher state limit exceeded")

    def add_bootstrap(prefix: str, began: int) -> None:
        """Retain one latest-start state for a shared source-prefix trie node."""
        previous_began = bootstrap.get(prefix)
        if previous_began is not None and previous_began >= began:
            return
        if previous_began is None:
            for character in prefix_children.get(prefix, ()):
                bootstrap_by_next[character].add(prefix)
            if prefix.endswith(" "):
                bootstrap_after_space.add(prefix)
        bootstrap[prefix] = began
        if len(live) + len(bootstrap) > _MAX_MATCH_STATES:
            raise ValueError("Publication matcher state limit exceeded")

    def clear_bootstrap() -> None:
        bootstrap.clear()
        bootstrap_by_next.clear()
        bootstrap_after_space.clear()

    def queue_addition(
        additions: dict[tuple[int, int], int],
        key: tuple[int, int],
        began: int,
    ) -> None:
        """Queue one deferred state without letting a token grow unbounded."""
        previous_began = additions.get(key)
        if previous_began is not None and previous_began >= began:
            return
        additions[key] = began
        if len(additions) > _MAX_MATCH_STATES:
            raise ValueError("Publication matcher state limit exceeded")

    def advance_noisy_body_run(
        body_index: int,
        position: int,
        token: str,
        cursor: int,
        began: int,
        additions: dict[tuple[int, int], int],
        maximum_run: int | None = None,
    ) -> bool:
        """Retain every exact body endpoint before token suffix noise."""
        nonlocal operations
        body = bodies[body_index]
        matched = 0
        while (
            position < len(body) and cursor < len(token)
            and (maximum_run is None or matched < maximum_run)
        ):
            operations += 1
            if operations > allowed_operations:
                raise ValueError("Publication matcher operation limit exceeded")
            if body[position] != token[cursor]:
                break
            position += 1
            cursor += 1
            matched += 1
            if position == len(body):
                return True
            queue_addition(additions, (body_index, position), began)
        return False

    def progress_bootstrap(
        token: str,
        offset: int,
        candidates,
        additions: dict[tuple[int, int], int],
        bootstrap_additions: dict[str, int],
    ) -> bool:
        """Advance every valid shared-prefix branch from one token offset."""
        nonlocal operations
        for prefix in candidates:
            began = bootstrap.get(prefix)
            if began is None:
                continue
            extended = prefix
            cursor = offset
            valid = True
            while cursor < len(token) and len(extended) < width:
                candidate = extended + token[cursor]
                operations += 1
                if operations > allowed_operations:
                    raise ValueError("Publication matcher operation limit exceeded")
                if candidate not in starts[len(candidate)]:
                    valid = False
                    break
                extended = candidate
                cursor += 1
                if extended in terminal_prefixes:
                    return True
                if allow_noise and len(extended) < width:
                    bootstrap_additions[extended] = max(
                        bootstrap_additions.get(extended, -1), began,
                    )
            if len(extended) < width:
                if (
                    not allow_noise and extended != prefix
                    and valid and cursor == len(token)
                ):
                    bootstrap_additions[extended] = max(
                        bootstrap_additions.get(extended, -1), began,
                    )
                continue
            fragment_length = len(token) - offset
            for body_index in starts[width].get(extended, ()):
                operations += 1
                if operations > allowed_operations:
                    raise ValueError("Publication matcher operation limit exceeded")
                body = bodies[body_index]
                remaining_length = len(body) - len(prefix)
                if remaining_length <= fragment_length:
                    if bounded_prefix(token, offset, body, len(prefix)):
                        return True
                if allow_noise and len(body) > width:
                    queue_addition(additions, (body_index, width), began)
                    if advance_noisy_body_run(
                        body_index, width, token, cursor, began, additions,
                    ):
                        return True
                elif (
                    remaining_length > fragment_length
                    and bounded_prefix(body, len(prefix), token, offset)
                ):
                    progressed = body_index, len(prefix) + fragment_length
                    queue_addition(additions, progressed, began)
        return False

    def noisy_initial_prefixes(token: str) -> tuple[str, ...]:
        """Return every shared source prefix embedded at every token offset."""
        nonlocal operations
        cached = noisy_prefix_cache.get(token)
        if cached is not None:
            operations += 1 + len(cached)
            if operations > allowed_operations:
                raise ValueError("Publication matcher operation limit exceeded")
            return cached
        matches = set()
        operations += len(token)
        if operations > allowed_operations:
            raise ValueError("Publication matcher operation limit exceeded")
        for offset, character in enumerate(token):
            if character not in starts[1]:
                continue
            shared_prefix = ""
            for size in range(1, min(width - 1, len(token) - offset) + 1):
                candidate = shared_prefix + token[offset + size - 1]
                operations += 1
                if operations > allowed_operations:
                    raise ValueError("Publication matcher operation limit exceeded")
                if candidate not in starts[size]:
                    break
                shared_prefix = candidate
                matches.add(shared_prefix)
        result = tuple(sorted(matches, key=lambda prefix: (len(prefix), prefix)))
        if len(token) <= 512 and len(noisy_prefix_cache) < 32768:
            noisy_prefix_cache[token] = result
        return result

    for token_index, item in enumerate(tokens):
        if isinstance(item, tuple):
            if (
                len(item) not in {2, 3, 4, 5} or not isinstance(item[0], str)
                or type(item[1]) is not bool
                or len(item) == 3 and type(item[2]) is not bool
                or len(item) in {4, 5} and (
                    type(item[2]) is not bool or type(item[3]) is not bool
                )
                or len(item) == 5 and type(item[4]) is not bool
            ):
                raise ValueError("Invalid ordered publication token")
            raw_token, structural = item[:2]
            allow_noise = item[2] if len(item) == 3 else True
            if len(item) >= 4:
                allow_noise, skippable = item[2:4]
            else:
                skippable = structural
        else:
            raw_token, structural, allow_noise, skippable = (
                item, False, True, False,
            )
        if raw_token == "\0" and not skippable:
            active = [defaultdict(set) for _ in range(width + 1)]
            terminal.clear()
            after_space.clear()
            live.clear()
            state_keys.clear()
            clear_bootstrap()
            continue
        additions: dict[tuple[int, int], int] = {}
        bootstrap_additions: dict[str, int] = {}
        variants = _canonical_token_variants(raw_token)
        has_empty_alternative = "" in variants
        has_space_alternative = " " in variants
        for token in variants:
            if not token:
                continue
            token_length = len(token)
            for index in short_starts:
                if bounded_contains(token, bodies[index]):
                    return finish(True)
            if allow_noise:
                for shared_prefix in noisy_initial_prefixes(token):
                    bootstrap_additions[shared_prefix] = max(
                        bootstrap_additions.get(shared_prefix, -1), token_index,
                    )
            for key in tuple(terminal):
                began = live.get(key)
                if began is None:
                    continue
                body_index, position = key
                body = bodies[body_index]
                remaining = body[position:]
                matched = (
                    bounded_contains(token, remaining)
                    if allow_noise
                    else bounded_prefix(token, 0, body, position)
                )
                if matched:
                    return finish(True)
            if not allow_noise and token.startswith(" ") and len(token) > 1:
                # Canonical whitespace is idempotent across adjacent parser
                # channels.  Only a state which already consumed the body's
                # single space may ignore one duplicated leading space.
                continuation = token[1:]
                size = min(width, len(continuation))
                continued_pool = active[size].get(continuation[:size], ())
                bootstrap_pool = bootstrap_by_next.get(continuation[0], ())
                operations += (
                    min(len(terminal), len(after_space))
                    + len(continued_pool) + len(bootstrap_pool)
                )
                if operations > allowed_operations:
                    raise ValueError(
                        "Publication matcher operation limit exceeded"
                    )
                terminal_candidates = tuple(terminal & after_space)
                continued_candidates = tuple(
                    key for key in continued_pool
                    if key in after_space
                )
                bootstrap_candidates = tuple(
                    prefix
                    for prefix in bootstrap_pool
                    if prefix in bootstrap_after_space
                )
                for key in terminal_candidates:
                    began = live.get(key)
                    if began is None:
                        continue
                    body_index, position = key
                    if bounded_prefix(
                        continuation, 0, bodies[body_index], position,
                    ):
                        return finish(True)
                for key in continued_candidates:
                    began = live.get(key)
                    if began is None:
                        continue
                    body_index, position = key
                    body = bodies[body_index]
                    remaining_length = len(body) - position
                    if remaining_length <= len(continuation):
                        if bounded_prefix(continuation, 0, body, position):
                            return finish(True)
                    elif bounded_prefix(body, position, continuation):
                        queue_addition(
                            additions,
                            (body_index, position + len(continuation)),
                            began,
                        )
                if bootstrap_candidates and progress_bootstrap(
                    continuation, 0, bootstrap_candidates,
                    additions, bootstrap_additions,
                ):
                    return finish(True)
            main_end = max(0, token_length - width + 1)
            for offset in range(main_end):
                prefix = token[offset:offset + width]
                start_candidates = starts[width].get(prefix, ())
                short_candidates = (
                    tuple(active[1].get(token[offset], ()))
                    if allow_noise and live else ()
                )
                active_candidates = (
                    tuple(active[width].get(prefix, ()))
                    if allow_noise or offset == 0 else ()
                )
                bootstrap_candidates = (
                    bootstrap_by_next.get(token[offset], ())
                    if bootstrap and (allow_noise or offset == 0) else ()
                )
                operations += (
                    1 + len(start_candidates) + len(short_candidates)
                    + len(active_candidates)
                    + len(bootstrap_candidates)
                )
                if operations > allowed_operations:
                    raise ValueError("Publication matcher operation limit exceeded")
                fragment_length = token_length - offset
                # A noisy carrier may contain only one to seven useful source
                # characters.  Index by the next required character, advance
                # each established branch by its maximal contiguous run, and
                # defer all states so one token cannot consume itself twice.
                for key in short_candidates:
                    began = live.get(key)
                    if began is None:
                        continue
                    body_index, position = key
                    if advance_noisy_body_run(
                        body_index, position, token, offset, began,
                        additions, maximum_run=width,
                    ):
                        return finish(True)
                for body_index in start_candidates:
                    body = bodies[body_index]
                    if len(body) <= fragment_length:
                        if bounded_prefix(token, offset, body):
                            return finish(True)
                    if allow_noise and len(body) > width:
                        key = body_index, width
                        queue_addition(additions, key, token_index)
                        if advance_noisy_body_run(
                            body_index, width, token, offset + width,
                            token_index, additions,
                        ):
                            return finish(True)
                    elif (
                        len(body) > fragment_length
                        and bounded_prefix(body, 0, token, offset)
                    ):
                        key = body_index, fragment_length
                        queue_addition(additions, key, token_index)
                for key in active_candidates:
                    began = live.get(key)
                    if began is None:
                        continue
                    body_index, position = key
                    body = bodies[body_index]
                    remaining_length = len(body) - position
                    if remaining_length <= fragment_length:
                        if bounded_prefix(token, offset, body, position):
                            return finish(True)
                    if allow_noise and remaining_length > width:
                        progressed = body_index, position + width
                        queue_addition(additions, progressed, began)
                        if advance_noisy_body_run(
                            body_index, position + width, token,
                            offset + width, began, additions,
                        ):
                            return finish(True)
                    elif (
                        remaining_length > fragment_length
                        and bounded_prefix(body, position, token, offset)
                    ):
                        progressed = (body_index, position + fragment_length)
                        queue_addition(additions, progressed, began)
                if bootstrap_candidates and progress_bootstrap(
                    token, offset, bootstrap_candidates,
                    additions, bootstrap_additions,
                ):
                    return finish(True)
            for offset in range(main_end, token_length):
                fragment = token[offset:]
                fragment_length = len(fragment)
                size = fragment_length
                prefix = fragment
                short_candidates = (
                    tuple(active[1].get(token[offset], ()))
                    if allow_noise and live else ()
                )
                active_candidates = (
                    tuple(active[size].get(prefix, ()))
                    if allow_noise or offset == 0 else ()
                )
                bootstrap_candidates = (
                    bootstrap_by_next.get(token[offset], ())
                    if bootstrap and (allow_noise or offset == 0) else ()
                )
                operations += (
                    1 + len(short_candidates) + len(active_candidates)
                    + len(bootstrap_candidates)
                )
                if operations > allowed_operations:
                    raise ValueError("Publication matcher operation limit exceeded")
                for key in short_candidates:
                    began = live.get(key)
                    if began is None:
                        continue
                    body_index, position = key
                    if advance_noisy_body_run(
                        body_index, position, token, offset, began,
                        additions, maximum_run=width,
                    ):
                        return finish(True)
                for key in active_candidates:
                    began = live.get(key)
                    if began is None:
                        continue
                    body_index, position = key
                    body = bodies[body_index]
                    remaining_length = len(body) - position
                    if remaining_length <= fragment_length:
                        if bounded_prefix(token, offset, body, position):
                            return finish(True)
                    elif bounded_prefix(body, position, fragment):
                        progressed = (body_index, position + fragment_length)
                        additions[progressed] = max(
                            additions.get(progressed, -1), began,
                        )
                if bootstrap_candidates and progress_bootstrap(
                    token, offset, bootstrap_candidates, additions,
                    bootstrap_additions,
                ):
                    return finish(True)
                if size < width and prefix in starts[size]:
                    bootstrap_additions[prefix] = max(
                        bootstrap_additions.get(prefix, -1), token_index,
                    )
        if not structural and not skippable:
            if has_empty_alternative:
                # A CSS line continuation (including nested URL/HTML forms)
                # contributes no parser-channel text under one valid decoding.
                # Retain only the frozen pre-token states; additions from other
                # variants remain deferred until after this token is cleared.
                operations += len(live) + len(bootstrap)
                if operations > allowed_operations:
                    raise ValueError(
                        "Publication matcher operation limit exceeded"
                    )
                for key, began in live.items():
                    queue_addition(additions, key, began)
                for prefix, began in bootstrap.items():
                    bootstrap_additions[prefix] = max(
                        bootstrap_additions.get(prefix, -1), began,
                    )
            elif has_space_alternative:
                operations += len(after_space) + len(bootstrap_after_space)
                if operations > allowed_operations:
                    raise ValueError(
                        "Publication matcher operation limit exceeded"
                    )
                for key in after_space:
                    began = live.get(key)
                    if began is not None:
                        queue_addition(additions, key, began)
                for prefix in bootstrap_after_space:
                    began = bootstrap.get(prefix)
                    if began is not None:
                        bootstrap_additions[prefix] = max(
                            bootstrap_additions.get(prefix, -1), began,
                        )
            if live:
                live.clear()
                state_keys.clear()
                terminal.clear()
                after_space.clear()
                active = [defaultdict(set) for _ in range(width + 1)]
            if bootstrap:
                clear_bootstrap()

        for (body_index, position), began in additions.items():
            add_state((body_index, position, began))
        for prefix, began in bootstrap_additions.items():
            add_bootstrap(prefix, began)
    return finish(False)


@lru_cache(maxsize=4)
def _ordered_pattern_automaton(pattern_indices: tuple[int, ...]):
    """Compile the fixed publication regexes to a Thompson NFA."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        import sre_parse

    edges: list[list[tuple[tuple, int]]] = []
    epsilon: list[list[int]] = []

    def new_state() -> int:
        if len(edges) >= _MAX_MATCH_STATES:
            raise ValueError("Publication pattern automaton state limit exceeded")
        edges.append([])
        epsilon.append([])
        return len(edges) - 1

    def compile_sequence(sequence, flags: int) -> tuple[int, int]:
        start = new_state()
        cursor = start
        for operation, argument in sequence:
            item_start, item_end = compile_item(operation, argument, flags)
            epsilon[cursor].append(item_start)
            cursor = item_end
        return start, cursor

    def compile_item(operation, argument, flags: int) -> tuple[int, int]:
        start, end = new_state(), new_state()
        if operation is sre_parse.LITERAL:
            edges[start].append((("literal", argument, bool(flags & re.I)), end))
        elif operation is sre_parse.NOT_LITERAL:
            edges[start].append((("not-literal", argument, bool(flags & re.I)), end))
        elif operation is sre_parse.ANY:
            edges[start].append((("any", bool(flags & re.S)), end))
        elif operation is sre_parse.CATEGORY:
            edges[start].append((("category", argument), end))
        elif operation is sre_parse.IN:
            negate = bool(argument and argument[0][0] is sre_parse.NEGATE)
            members = tuple(argument[1:] if negate else argument)
            edges[start].append((
                ("set", negate, members, bool(flags & re.I)), end,
            ))
        elif operation is sre_parse.SUBPATTERN:
            _group, add_flags, delete_flags, child = argument
            child_flags = (flags | add_flags) & ~delete_flags
            child_start, child_end = compile_sequence(child, child_flags)
            epsilon[start].append(child_start)
            epsilon[child_end].append(end)
        elif operation is sre_parse.BRANCH:
            _none, branches = argument
            for branch in branches:
                branch_start, branch_end = compile_sequence(branch, flags)
                epsilon[start].append(branch_start)
                epsilon[branch_end].append(end)
        elif operation in {sre_parse.MAX_REPEAT, sre_parse.MIN_REPEAT}:
            minimum, maximum, child = argument
            cursor = start
            for _ in range(minimum):
                child_start, child_end = compile_sequence(child, flags)
                epsilon[cursor].append(child_start)
                cursor = child_end
            epsilon[cursor].append(end)
            if maximum is sre_parse.MAXREPEAT:
                child_start, child_end = compile_sequence(child, flags)
                epsilon[cursor].append(child_start)
                epsilon[child_end].append(cursor)
            else:
                for _ in range(maximum - minimum):
                    child_start, child_end = compile_sequence(child, flags)
                    epsilon[cursor].append(child_start)
                    cursor = child_end
                    epsilon[cursor].append(end)
        elif operation in {sre_parse.AT, sre_parse.ASSERT, sre_parse.ASSERT_NOT}:
            # Word boundaries were removed for semantic projections. Ignoring
            # the one negative path lookbehind is conservative: it can only
            # reject more reconstructed strings, never miss a private one.
            epsilon[start].append(end)
        else:
            raise ValueError("Unsupported publication pattern operation")
        return start, end

    root = new_state()
    terminals = set()
    boundary_starts = set()
    free_starts = set()
    boundary_terminals = set()
    for index in pattern_indices:
        pattern = _SEMANTIC_PATTERNS[index]
        parsed = list(sre_parse.parse(pattern.pattern, pattern.flags))
        starts_on_boundary = bool(
            parsed and parsed[0][0] is sre_parse.AT
            and parsed[0][1] is sre_parse.AT_BOUNDARY
        )
        ends_on_boundary = bool(
            parsed and parsed[-1][0] is sre_parse.AT
            and parsed[-1][1] is sre_parse.AT_BOUNDARY
        )
        if starts_on_boundary:
            parsed = parsed[1:]
        if ends_on_boundary:
            parsed = parsed[:-1]
        pattern_start, pattern_end = compile_sequence(parsed, pattern.flags)
        epsilon[root].append(pattern_start)
        terminals.add(pattern_end)
        (boundary_starts if starts_on_boundary else free_starts).add(
            pattern_start
        )
        if ends_on_boundary:
            boundary_terminals.add(pattern_end)
    if 0 in pattern_indices:
        parsed = sre_parse.parse(
            _PRIVATE_KEY_HEADER.pattern, _PRIVATE_KEY_HEADER.flags,
        )
        header_start, header_end = compile_sequence(
            parsed, _PRIVATE_KEY_HEADER.flags,
        )
        epsilon[root].append(header_start)
        terminals.add(header_end)
        free_starts.add(header_start)
    if 10 in pattern_indices:
        parsed = list(sre_parse.parse(
            _ORDERED_FIELD_ASSIGNMENT.pattern,
            _ORDERED_FIELD_ASSIGNMENT.flags,
        ))
        starts_on_boundary = bool(
            parsed and parsed[0][0] is sre_parse.ASSERT_NOT
        )
        if starts_on_boundary:
            parsed = parsed[1:]
        assignment_start, assignment_end = compile_sequence(
            parsed, _ORDERED_FIELD_ASSIGNMENT.flags,
        )
        epsilon[root].append(assignment_start)
        terminals.add(assignment_end)
        (boundary_starts if starts_on_boundary else free_starts).add(
            assignment_start
        )

    closures = []
    for state in range(len(edges)):
        found = {state}
        stack = [state]
        while stack:
            current = stack.pop()
            for target in epsilon[current]:
                if target not in found:
                    found.add(target)
                    stack.append(target)
        closures.append(frozenset(found))
    return (
        tuple(tuple(item) for item in edges), tuple(closures),
        frozenset(terminals), frozenset(boundary_terminals),
        frozenset(
            state for start in free_starts for state in closures[start]
        ),
        frozenset(
            state for start in boundary_starts for state in closures[start]
        ),
    )


def _ordered_pattern_member(rule: tuple, character: str) -> bool:
    """Evaluate one immutable NFA character predicate."""
    import sre_parse
    kind = rule[0]
    if kind == "any":
        return rule[1] or character != "\n"
    if kind in {"literal", "not-literal"}:
        expected = chr(rule[1])
        equal = (
            character.casefold() == expected.casefold()
            if rule[2] else character == expected
        )
        return equal if kind == "literal" else not equal
    if kind == "category":
        category = rule[1]
        if category is sre_parse.CATEGORY_WORD:
            return character.isalnum() or character == "_"
        if category is sre_parse.CATEGORY_DIGIT:
            return character.isdigit()
        if category is sre_parse.CATEGORY_SPACE:
            return character.isspace()
        raise ValueError("Unsupported publication pattern category")
    _kind, negate, members, ignore_case = rule
    matched = False
    candidates = {character}
    if ignore_case:
        candidates.update({character.lower(), character.upper(), character.casefold()})
    for operation, argument in members:
        if operation is sre_parse.LITERAL:
            matched = any(value == chr(argument) for value in candidates)
        elif operation is sre_parse.RANGE:
            matched = any(
                len(value) == 1 and argument[0] <= ord(value) <= argument[1]
                for value in candidates
            )
        elif operation is sre_parse.CATEGORY:
            matched = _ordered_pattern_member(("category", argument), character)
        else:
            raise ValueError("Unsupported publication pattern set operation")
        if matched:
            break
    return not matched if negate else matched


def _ordered_pattern_violation(
    tokens: list[
        str | tuple[str, bool] | tuple[str, bool, bool]
        | tuple[str, bool, bool, bool] | tuple[str, bool, bool, bool, bool]
    ],
    publication_budget: list[int],
    file_budget: list[int],
    *,
    pattern_indices: tuple[int, ...] = _DISTINCTIVE_ORDERED_PATTERNS,
    start_within_skippable: bool = True,
    select_substrings: bool = True,
    respect_startability: bool = False,
    absolute_path_boundary: bool = False,
) -> bool:
    """Run fixed security patterns across selectable HTML parser channels."""
    (
        edges, closures, terminals, boundary_terminals,
        free_start_states, boundary_start_states,
    ) = _ordered_pattern_automaton(pattern_indices)
    allowed = min(publication_budget[0], file_budget[0])
    operations = 0
    states: set[int] = set()

    def finish(result: bool) -> bool:
        publication_budget[0] -= operations
        file_budget[0] -= operations
        return result

    def step(
        current: set[int], character: str, allow_start: bool,
        boundary_start: bool,
    ) -> tuple[bool, bool, set[int]]:
        nonlocal operations
        source = set(current)
        if allow_start:
            source.update(free_start_states)
            if boundary_start:
                source.update(boundary_start_states)
        additions: set[int] = set()
        for state in source:
            for rule, target in edges[state]:
                operations += 1
                if operations > allowed:
                    raise _PublicationMatchLimit(
                        "Publication matcher operation limit exceeded"
                    )
                if _ordered_pattern_member(rule, character):
                    additions.update(closures[target])
        reached = additions & terminals
        return (
            bool(reached - boundary_terminals),
            bool(reached & boundary_terminals), additions,
        )

    for item in tokens:
        if isinstance(item, tuple):
            if (
                len(item) not in {2, 3, 4, 5} or not isinstance(item[0], str)
                or type(item[1]) is not bool
                or len(item) == 3 and type(item[2]) is not bool
                or len(item) in {4, 5} and (
                    type(item[2]) is not bool or type(item[3]) is not bool
                )
                or len(item) == 5 and type(item[4]) is not bool
            ):
                raise ValueError("Invalid ordered publication token")
            raw_token, structural = item[:2]
            skippable = item[3] if len(item) >= 4 else structural
            startable = not (
                respect_startability and len(item) == 5 and not item[4]
            )
        else:
            raw_token, skippable, startable = item, False, True
        if raw_token == "\0" and not skippable:
            states.clear()
            continue
        next_states = set(states) if skippable else set()
        for token in _canonical_token_variants(raw_token):
            active = set(states)
            endpoints = set()
            pending_boundary = False
            if not token:
                next_states.update(active)
            for character_index, character in enumerate(token):
                if pending_boundary and not (
                    character.isalnum() or character == "_"
                ):
                    return finish(True)
                pending_boundary = False
                boundary_startable = not (
                    absolute_path_boundary and character_index > 0
                    and (
                        token[character_index - 1].isalnum()
                        or token[character_index - 1] in "_:/."
                    )
                )
                # Frozen states are reintroduced only for nonvisible tokens,
                # permitting one contiguous substring plus inert prefix/suffix.
                previous = token[character_index - 1] if character_index else ""
                word_boundary = (
                    (character.isalnum() or character == "_")
                    != (previous.isalnum() or previous == "_")
                )
                hit, pending_boundary, active = step(
                    active | states
                    if skippable and select_substrings else active,
                    character,
                    startable and boundary_startable and (
                        not skippable or start_within_skippable
                        or character_index == 0
                    ),
                    word_boundary,
                )
                if hit:
                    return finish(True)
                if skippable and select_substrings:
                    endpoints.update(active)
            if pending_boundary:
                return finish(True)
            next_states.update(
                endpoints if skippable and select_substrings else active
            )
        states = next_states
    return finish(False)


def _inventory_parts(data: bytes) -> list[str]:
    """Validate a Sphinx v2 inventory and expose its decompressed text to scans."""
    import zlib

    sections = data.split(b"\n", 4)
    if len(sections) != 5 or sections[0] != b"# Sphinx inventory version 2":
        raise ValueError("Invalid Sphinx inventory header")
    if sections[3] != b"# The remainder of this file is compressed using zlib.":
        raise ValueError("Invalid Sphinx inventory encoding")
    decompressor = zlib.decompressobj()
    body = decompressor.decompress(sections[4], 50_000_001)
    if len(body) > 50_000_000 or decompressor.unconsumed_tail:
        raise ValueError("Sphinx inventory exceeds publication limit")
    body += decompressor.flush()
    if len(body) > 50_000_000 or not decompressor.eof or decompressor.unused_data:
        raise ValueError("Invalid Sphinx inventory payload")
    return [b"\n".join(sections[:4]).decode("utf-8"), body.decode("utf-8")]


def _source_bodies(root: Path | None) -> tuple[set[str], list[str]]:
    """Keep exact/normalized raw-body evidence locally, never in the artifact.

    Complete-body matching catches source copied into an innocent download or
    rendered with syntax-highlighting spans. It deliberately does not flag
    partial snippets that can also be legitimate AST-derived docstrings.
    Authored Sphinx presentation files are the explicit publication inputs.
    """
    if root is None:
        return set(), []
    from wiki_repository import _kind, repository_files
    hashes, bodies = set(), set()
    root = root.resolve()
    for path in repository_files(root):
        relative = path.relative_to(root)
        if relative.parts[:2] == ("docs", "wiki") or relative.as_posix() in {"docs/index.rst", "doc/index-hypertagging.rst"}:
            continue
        _, readable = _kind(path)
        if not readable:
            continue
        if path.stat().st_size > _MAX_SOURCE_TEXT_BYTES:
            # A digest catches a byte-identical download but cannot catch a
            # highlighted/markup-split copy. Refuse publication instead of
            # silently weakening complete-body protection for a large source.
            raise ValueError("Readable source exceeds raw-body scan limit")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        if path.stat().st_size:
            hashes.add(digest.hexdigest())
        try:
            body = _canonical_body(path.read_text(encoding="utf-8"))
        except (ValueError, UnicodeError, OSError):
            raise ValueError("Unreadable classified text source") from None
        if body:
            bodies.add(body)
    return hashes, sorted(bodies, key=len)


def _trusted_vendor_hashes() -> dict[str, set[str]]:
    """Authenticate byte-identical Sphinx static assets emitted by this install.

    Sphinx 3 copied jQuery/Underscore while newer releases ship a smaller set.
    The build already executes this installed Sphinx distribution, so its exact
    static bytes are within the same trust boundary.  Name *and* digest must
    match; authored or modified assets never receive this exception.
    """
    hashes: dict[str, set[str]] = {
        name: set(digests) for name, digests in _SAFE_GENERATED_JS.items()
    }
    try:
        import sphinx
        from jinja2.sandbox import SandboxedEnvironment
        themes = Path(sphinx.__file__).parent / "themes"
        static = themes / "basic/static"

        def add(name: str, data: bytes) -> None:
            hashes.setdefault(name, set()).add(hashlib.sha256(data).hexdigest())

        for asset in themes.glob("*/static/*"):
            if (
                asset.suffix in {".css", ".js"}
                or asset.suffix.lower() in _BINARY_ASSETS
            ) and asset.is_file() and not asset.is_symlink():
                add(asset.name, asset.read_bytes())

        # The classic theme renders these templates with a small, fixed set of
        # public options from docs/wiki/conf.py.  Recreate only that allowlisted
        # configuration here: using arbitrary builder context would let an
        # authored theme option smuggle a private value into a trusted asset.
        from sphinx.jinja2glue import _tobool, _todim
        css_environment = SandboxedEnvironment()
        css_environment.filters.update(tobool=_tobool, todim=_todim)
        css_context = {
            "theme_sidebarwidth": "270px",
            "theme_body_min_width": "360",
            "theme_body_max_width": "1180px",
            "theme_rightsidebar": "false",
            "theme_stickysidebar": "false",
            "theme_collapsiblesidebar": "false",
            "theme_externalrefs": "false",
            "theme_footerbgcolor": "#11303d",
            "theme_footertextcolor": "#ffffff",
            "theme_sidebarbgcolor": "#1c4e63",
            "theme_sidebarbtncolor": "#3c6e83",
            "theme_sidebartextcolor": "#ffffff",
            "theme_sidebarlinkcolor": "#98dbcc",
            "theme_relbarbgcolor": "#133f52",
            "theme_relbartextcolor": "#ffffff",
            "theme_relbarlinkcolor": "#ffffff",
            "theme_bgcolor": "#ffffff",
            "theme_textcolor": "#000000",
            "theme_headbgcolor": "#f2f2f2",
            "theme_headtextcolor": "#20435c",
            "theme_headlinkcolor": "#c60f0f",
            "theme_linkcolor": "#355f7c",
            "theme_visitedlinkcolor": "#551a8b",
            "theme_codebgcolor": "unset",
            "theme_codetextcolor": "unset",
            "theme_bodyfont": "sans-serif",
            "theme_headfont": "'Trebuchet MS', sans-serif",
        }
        for theme_name in ("basic", "classic"):
            template_path = (
                themes / theme_name / "static" / f"{theme_name}.css_t"
            )
            if template_path.is_file() and not template_path.is_symlink():
                rendered = css_environment.from_string(
                    template_path.read_text(encoding="utf-8")
                ).render(css_context)
                add(f"{theme_name}.css", rendered.encode("utf-8"))

        from sphinx.highlighting import PygmentsBridge
        add(
            "pygments.css",
            PygmentsBridge("html", "sphinx").get_stylesheet().encode("utf-8"),
        )
        template = next((static / name for name in ("language_data.js.jinja", "language_data.js_t") if (static / name).is_file()), None)
        if template is not None:
            try:
                # Sphinx 3 can construct this without a build environment and
                # uses jsdump formatting in the generated asset.
                from sphinx.search import IndexBuilder
                context = IndexBuilder(None, "en", {}, "").context_for_searchtool()
            except AttributeError:
                # Sphinx 7+ requires a populated BuildEnvironment in its index
                # constructor.  English's fixed language data is sufficient for
                # the no-scoring/no-custom-splitter configuration used here.
                from sphinx.search.en import SearchEnglish
                language = SearchEnglish({})
                context = {
                    "search_language_stop_words": json.dumps(sorted(language.stopwords)),
                    "search_language_stemming_code": language.js_stemmer_code,
                    "search_scorer_tool": "", "search_word_splitter_code": "",
                }
            data = SandboxedEnvironment().from_string(template.read_text()).render(context).encode("utf-8")
            add("language_data.js", data)
    except (ImportError, OSError, AttributeError):
        # Directly copied assets already collected above remain authenticated if
        # only optional language rendering is unavailable.
        pass
    return hashes


def validate_artifact(
    directory: Path, root: Path | None = None, *,
    generated_projection: bool = False,
    renderer_output: bool = False,
) -> dict:
    """Reject private fields, encoded identifiers, filenames and raw source copies.

    JSON, HTML entities/attributes, text and the Sphinx search index are decoded.
    Only rule IDs and opaque location hashes enter diagnostics. Checks remain
    active under optimized Python and do not depend on assertion statements.
    """
    if not directory.is_dir() or directory.is_symlink():
        raise ValueError("Privacy scan requires a regular artifact directory")
    if any(parent.is_symlink() for parent in directory.absolute().parents):
        raise ValueError("Privacy scan refuses filesystem aliases")
    trusted_projection = generated_projection or renderer_output
    publication_match_budget = [_MAX_PUBLICATION_MATCH_OPERATIONS]
    deny = sensitive_literals(root) if root is not None else set()
    literal_values = sorted(deny, key=lambda item: (-len(item), item))
    ordered_literal_values = sorted({
        canonical for literal in literal_values
        if (canonical := _canonical_body(literal))
    }, key=lambda item: (-len(item), item))
    literal_matcher = (
        _literal_automaton(ordered_literal_values, publication_match_budget)
        if ordered_literal_values else None
    )
    raw_hashes, raw_bodies = _source_bodies(root)
    # The ordered parser-channel matcher is the expensive bounded pass.  Scan
    # source bodies and deny-listed structured values together so adding the
    # literal policy cannot silently double the publication operation budget.
    # dict preserves deterministic source-first ordering while deduplicating a
    # literal that is itself already a complete canonical source body.
    ordered_source_material = list(dict.fromkeys([
        *raw_bodies, *ordered_literal_values,
    ]))
    ordered_source_prefix_index = (
        _body_prefix_index(ordered_source_material)
        if ordered_source_material else None
    )
    raw_prefix_width = min(32, min(map(len, raw_bodies), default=32))
    raw_prefixes: dict[str, list[str]] = {}
    for body in raw_bodies:
        raw_prefixes.setdefault(body[:raw_prefix_width], []).append(body)
    vendor_hashes = _trusted_vendor_hashes()
    errors, scanned = [], 0
    textual_suffixes = {".html", ".json", ".txt", ".rst", ".js", ".css"}
    publication_paths = []
    for path in directory.rglob("*"):
        publication_paths.append(path)
        if len(publication_paths) > _MAX_PUBLICATION_ENTRIES:
            raise ValueError("Publication entry limit exceeded")
    publication_bytes = 0
    literal_file_budget = [_MAX_MATCH_OPERATIONS]
    raw_body_file_budget = [_MAX_MATCH_OPERATIONS]
    pattern_file_budget = [_MAX_MATCH_OPERATIONS]

    def inspect(value, *, location, trusted_vendor=False,
                field_assignments=True, html_markup=False,
                literal_scan=True, semantic_projection=False):
        rules = [] if trusted_vendor else violations(
            value, field_assignments=field_assignments, html_markup=html_markup,
            semantic_projection=semantic_projection,
        )
        if not trusted_vendor and literal_scan and literal_matcher is not None:
            if any(
                _literal_automaton_contains(
                    literal_matcher, variant, publication_match_budget,
                    literal_file_budget,
                )
                for variant in _canonical_variants(value)
            ):
                rules.append("private-source-literal")
        if rules:
            errors.append(location + ":" + ",".join(sorted(set(rules))))

    for path in sorted(publication_paths):
        literal_file_budget = [_MAX_MATCH_OPERATIONS]
        raw_body_file_budget = [_MAX_MATCH_OPERATIONS]
        pattern_file_budget = [_MAX_MATCH_OPERATIONS]
        rel = path.relative_to(directory)
        location = hashlib.sha256(rel.as_posix().encode("utf-8")).hexdigest()[:12]
        inspect(rel.as_posix(), location=location)
        if path.is_symlink():
            errors.append(location + ":filesystem-alias")
            continue
        if set(rel.parts) & {"evidence", "sources", "_sources", "_modules"}:
            errors.append(location + ":forbidden-source-directory")
            continue
        if not path.is_file():
            continue
        file_stat = path.stat()
        publication_bytes += file_stat.st_size
        if (
            file_stat.st_size > _MAX_PUBLICATION_FILE_BYTES
            or publication_bytes > _MAX_PUBLICATION_BYTES
        ):
            errors.append(location + ":publication-resource-limit")
            continue
        if file_stat.st_nlink != 1:
            errors.append(location + ":filesystem-alias")
            continue
        suffix = path.suffix.lower()
        allowed_binary = suffix in _BINARY_ASSETS and "_static" in rel.parts
        special_file = path.name in {"objects.inv", ".nojekyll", ".buildinfo"}
        if suffix not in textual_suffixes and not allowed_binary and not special_file:
            errors.append(location + ":forbidden-raw-download")
            continue
        try:
            data = path.read_bytes()
            raw_token_groups = []
            raw_ordered_tokens = []
            broad_ordered_tokens = []
            assignment_ordered_tokens = []
            path_ordered_tokens = []
            ordered_pattern_indices = _DISTINCTIVE_ORDERED_PATTERNS
            tag_match_budget = [_MAX_MATCH_OPERATIONS]
            digest = hashlib.sha256(data).hexdigest()
            trusted_vendor = (
                "_static" in rel.parts
                and digest in vendor_hashes.get(path.name, set())
            )
            if suffix == ".js" and path.name != "searchindex.js" and not trusted_vendor:
                errors.append(location + ":untrusted-active-script")
            if digest in raw_hashes and not trusted_vendor:
                errors.append(location + ":raw-source-body")
                continue
            if allowed_binary:
                if not trusted_vendor:
                    errors.append(location + ":untrusted-binary-asset")
                continue
            if path.name == "objects.inv":
                parts = _inventory_parts(data)
                combined = "\n".join(parts)
                parts = [combined]
                raw_body_projections = [combined]
                scanned += 1
            else:
                payload = data.decode("utf-8")
                scanned += 1
            if path.name == "objects.inv":
                pass
            elif suffix == ".html":
                parser = _PublishedHTML()
                parser.feed(payload)
                parser.close()
                (parser.raw_names, raw_token_groups, raw_ordered_tokens,
                 raw_markup_complete) = _raw_markup_channels(payload)
                # Broad rules (email, IP, host, assignment and path shapes)
                # are intentionally reconstructed inside the hidden channel.
                # Visible text is scanned by its own parser projection below;
                # mixing a link label with punctuation from an unrelated href
                # would manufacture values that neither channel exposes.
                broad_ordered_tokens = []
                for item in raw_ordered_tokens:
                    if item[0] in _PUBLIC_HTML_RECONSTRUCTION_RESETS:
                        # These fixed Sphinx/browser metadata fields are
                        # independent records, not one selectable value. A
                        # reset prevents two emitted viewport declarations
                        # from synthesizing checkpoint suffixes such as .pth.
                        broad_ordered_tokens.append(("\0", False, False))
                    elif (
                        len(item) in {4, 5} and item[0] != "="
                    ):
                        broad_ordered_tokens.append(item)
                    elif len(item) == 3 and not item[1]:
                        # Broad shapes are false-positive prone when fragments
                        # from unrelated document sections are combined. A
                        # non-matching visible sentinel resets that hidden
                        # channel while still permitting arbitrary selection
                        # within one adjacent markup region.
                        broad_ordered_tokens.append(("\0", False, False))
                assignment_ordered_tokens = _assignment_ordered_regions(
                    broad_ordered_tokens
                )
                path_ordered_tokens = broad_ordered_tokens
                raw_attribute_projection = "".join(
                    token
                    for group in raw_token_groups
                    for token_index, (token, structural, _allow_noise)
                    in enumerate(group)
                    if structural and token_index > 0
                )
                if not raw_markup_complete:
                    errors.append(location + ":unparseable-html-markup")
                if [name.lower() for name in parser.raw_names] != parser.names:
                    errors.append(location + ":unparseable-html-names")
                if not renderer_output:
                    for token_group in raw_token_groups:
                        if _ordered_tag_violation(
                            token_group, publication_match_budget,
                            tag_match_budget,
                        ):
                            errors.append(
                                location + ":private-field-assignment"
                            )
                if parser.private_field_value:
                    errors.append(location + ":private-field-value")
                if parser.private_path_value:
                    errors.append(location + ":private-pattern-4")
                if not renderer_output and "private-field-assignment" in violations(
                    "".join(parser.hidden), semantic_projection=True,
                ):
                    errors.append(location + ":private-field-assignment")
                if parser.duplicate_attribute:
                    errors.append(location + ":duplicate-html-attribute")
                if parser.active_content:
                    errors.append(location + ":untrusted-active-content")
                allowed_script_names = set(vendor_hashes) | {"searchindex.js"}
                for source in parser.script_sources:
                    parsed_source = urlsplit(source)
                    source_path = unquote(parsed_source.path)
                    if (
                        parsed_source.scheme or parsed_source.netloc
                        or source_path.startswith("/")
                        or Path(source_path).name not in allowed_script_names
                    ):
                        errors.append(location + ":untrusted-active-script")
                for action in parser.form_actions:
                    for variant in _runtime_url_variants(action):
                        action_path = urlsplit(variant).path
                        target = (
                            (path.parent / action_path).resolve()
                            if action_path else path.resolve()
                        )
                        if not target.is_relative_to(directory.resolve()):
                            errors.append(
                                location + ":untrusted-active-content"
                            )
                            break
                markup_projection = re.sub(
                    r"<[^>]*>", "",
                    payload.replace("<!--", "").replace("-->", ""),
                )
                raw_body_projections = [
                    payload, markup_projection,
                    "".join(parser.parts), "".join(parser.content),
                    "".join(parser.names), "".join(parser.raw_names),
                    raw_attribute_projection, "".join(parser.visible),
                    "".join(parser.semantic),
                    "".join(parser.semantic_names_visible),
                    "".join(parser.semantic_values_visible),
                ]
                # The raw HTML covers every contiguous byte sequence. Scan the
                # reconstructed content channel once more for values split by
                # markup, without repeatedly scanning tag/attribute-name noise.
                parts = [
                    payload, ";\n".join(parser.content), "".join(parser.content),
                ]
                combined = "".join(parser.visible)
                if renderer_output:
                    # The build entry point validates every authored source and
                    # forbids raw/include/file directives before Sphinx parses
                    # it; generated sources pass their own fresh-stage privacy
                    # gate. Keep direct, visible, parsed-field and active-content
                    # checks here, while avoiding a combinatorial reinterpretation
                    # of deterministic theme/index markup across unrelated tags.
                    raw_ordered_tokens = []
                    broad_ordered_tokens = []
                    assignment_ordered_tokens = []
                    path_ordered_tokens = []
                    raw_body_projections = [payload, combined]
                    parts = [payload, "".join(parser.content)]
            elif suffix == ".json":
                decoded = _strict_json(payload)
                if _contains_private_fields(decoded):
                    errors.append(location + ":private-field-value")
                scalar_parts = list(_strings(decoded))
                # Scan actual scalar values independently too, so assignment
                # syntax embedded inside a JSON string cannot hide behind the
                # surrounding serialized topology.
                for scalar in scalar_parts:
                    inspect(
                        scalar, location=location,
                        literal_scan=trusted_projection,
                    )
                    if trusted_projection and raw_bodies and any(
                        _raw_body_contains(
                            raw_prefixes, raw_prefix_width, variant,
                            publication_match_budget, raw_body_file_budget,
                        )
                        for variant in _canonical_variants(scalar)
                    ):
                        errors.append(location + ":raw-source-body")
                separated_projection = "\n".join(scalar_parts)
                key_projection = "".join(_keys(decoded))
                value_projection = "".join(_values(decoded))
                mixed_projection = "".join(scalar_parts)
                # Field semantics are checked on the parsed tree above. Scan
                # decoded scalar strings separated by newlines so JSON syntax
                # cannot turn a type annotation inside a string into a field.
                parts = [separated_projection]
                # Mixed, key-only and value-only streams cover every scalar and
                # the distinct reconstructions a data consumer can perform;
                # the payload preserves complete structured source fragments.
                # Individual scalars are redundant subsets of these streams.
                raw_body_projections = [
                    payload, separated_projection, mixed_projection,
                    key_projection, value_projection,
                ]
                combined = payload
                # Generator projections are produced in a fresh private stage,
                # then independently regenerated and byte-compared before
                # publication. Their multi-megabyte API arrays are not an
                # attacker-controlled selectable JSON channel. Every scalar,
                # key/value relation, source literal and raw-body projection
                # is still scanned below; only combinatorial cross-scalar NFA
                # reconstruction is omitted for this authenticated boundary.
                raw_ordered_tokens = (
                    [] if trusted_projection
                    else _structured_ordered_tokens(decoded)
                )
                broad_ordered_tokens = raw_ordered_tokens
                assignment_ordered_tokens = _assignment_ordered_regions(
                    raw_ordered_tokens
                )
                path_ordered_tokens = raw_ordered_tokens
            elif path.name == "searchindex.js":
                match = re.fullmatch(r"\s*Search\.setIndex\((.*)\);?\s*", payload, re.S)
                if not match:
                    raise ValueError("Invalid search index")
                try:
                    decoded = _strict_json(match.group(1))
                except json.JSONDecodeError:
                    # Sphinx 3 used its restricted JS-dump syntax; newer Sphinx
                    # emits JSON.  This parser is data-only and never evaluates
                    # JavaScript.
                    try:
                        from sphinx.util import jsdump
                    except ImportError as error:
                        raise ValueError("Unsupported search index encoding") from error
                    encoded = match.group(1)
                    decoded = jsdump.loads(encoded)
                    # The historical loader stops after the first complete root
                    # value.  Require its canonical encoder to reproduce every
                    # input byte so an ignored argument, operator or comment
                    # cannot carry unscanned publication content.
                    if jsdump.dumps(decoded) != encoded:
                        raise ValueError("Noncanonical legacy search index")
                if not _valid_search_index(decoded):
                    raise ValueError("Invalid search index schema")
                scalar_parts = list(_strings(decoded))
                # Search-index dictionaries have typed structural keys whose
                # integer document references are not credential assignments.
                # Scan each actual scalar independently for assignment syntax;
                # do not reinterpret the serialized index topology as one.
                for scalar in scalar_parts:
                    inspect(
                        scalar, location=location, literal_scan=False,
                    )
                separated_projection = "\n".join(scalar_parts)
                key_projection = "".join(_keys(decoded))
                value_projection = "".join(_values(decoded))
                mixed_projection = "".join(scalar_parts)
                parts = [separated_projection]
                raw_body_projections = [
                    payload, separated_projection, mixed_projection,
                    key_projection, value_projection,
                ]
                combined = payload
                raw_ordered_tokens = _search_index_ordered_tokens(decoded)
                # Related typed strings can still split distinctive tokens and
                # email addresses. Generic host/path/checkpoint patterns are
                # scanned within each scalar above; applying them across an
                # object triple turns public names such as Node + p4 into a
                # synthetic worker host or checkpoint filename.
                ordered_pattern_indices = (
                    *_DISTINCTIVE_ORDERED_PATTERNS, 5,
                )
                broad_ordered_tokens = []
                assignment_ordered_tokens = _assignment_ordered_regions(
                    raw_ordered_tokens
                )
                path_ordered_tokens = []
            else:
                parts = [payload]
                raw_body_projections = [payload]
                combined = payload
            if suffix == ".css":
                raw_ordered_tokens = _css_comment_tokens(payload)
                broad_ordered_tokens = raw_ordered_tokens
                path_ordered_tokens = raw_ordered_tokens
                if raw_ordered_tokens and any(
                    ":" in variant or "=" in variant
                    for token in raw_ordered_tokens
                    for variant in _decoded_variants(token[0])
                ):
                    assignment_ordered_tokens = raw_ordered_tokens
                if not trusted_vendor and _unsafe_css(payload):
                    errors.append(location + ":untrusted-active-content")
            if (
                not trusted_vendor
                and raw_ordered_tokens
                and _has_useful_cross_fragment_encoding(
                    raw_ordered_tokens, publication_match_budget,
                    pattern_file_budget,
                )
            ):
                errors.append(location + ":incomplete-encoded-fragment")
        except _PublicationMatchLimit:
            raise ValueError(
                "Publication matcher operation limit exceeded at " + location
            ) from None
        except (ValueError, UnicodeError, OSError):
            errors.append(location + ":unreadable-publication")
            continue
        if not trusted_vendor:
            ordered_hit = (
                _ordered_token_stream_contains(
                    ordered_source_material, raw_ordered_tokens,
                    publication_match_budget, ordered_source_prefix_index,
                )
                if ordered_source_material and raw_ordered_tokens else False
            )
            ordered_pattern_hit = (
                    _ordered_pattern_violation(
                        raw_ordered_tokens, publication_match_budget,
                        pattern_file_budget,
                        pattern_indices=ordered_pattern_indices,
                        respect_startability=True,
                    select_substrings=(suffix != ".json"),
                )
                if raw_ordered_tokens else False
            )
            ordered_assignment_hit = (
                _ordered_pattern_violation(
                    assignment_ordered_tokens, publication_match_budget,
                    pattern_file_budget,
                    pattern_indices=(10,),
                    respect_startability=True,
                    select_substrings=(suffix != ".json"),
                )
                if assignment_ordered_tokens else False
            )
            substring_pattern_hit = (
                _ordered_pattern_violation(
                    broad_ordered_tokens, publication_match_budget,
                    pattern_file_budget,
                    pattern_indices=_SUBSTRING_ORDERED_PATTERNS,
                    respect_startability=True,
                    select_substrings=(suffix != ".json"),
                )
                if broad_ordered_tokens else False
            )
            absolute_path_hit = substring_pattern_hit or (
                _ordered_pattern_violation(
                    path_ordered_tokens, publication_match_budget,
                    pattern_file_budget,
                    pattern_indices=_ABSOLUTE_PATH_ORDERED_PATTERNS,
                    respect_startability=True,
                    select_substrings=(suffix != ".json"),
                )
                if path_ordered_tokens else False
            )
            broad_pattern_hit = absolute_path_hit or (
                _ordered_pattern_violation(
                    path_ordered_tokens, publication_match_budget,
                    pattern_file_budget,
                    pattern_indices=_PATH_FRAGMENT_ORDERED_PATTERNS,
                    respect_startability=True,
                    select_substrings=(suffix != ".json"),
                )
                if path_ordered_tokens else False
            )
        else:
            ordered_hit = False
            ordered_pattern_hit = False
            ordered_assignment_hit = False
            broad_pattern_hit = False
        if ordered_assignment_hit:
            errors.append(location + ":private-field-assignment")
        if ordered_pattern_hit or broad_pattern_hit:
            errors.append(location + ":private-pattern-ordered")
        if not trusted_vendor:
            for projection_index, projection in enumerate(
                [] if suffix == ".json" or path.name == "searchindex.js"
                else raw_body_projections
            ):
                inspect(
                    projection, location=location,
                    html_markup=(suffix == ".html" and projection_index == 0),
                    field_assignments=(
                        projection_index == 0
                        and path.name != "searchindex.js"
                        and suffix != ".json"
                    ),
                    literal_scan=False,
                    # Boundary-free semantic projections are meaningful only
                    # for parser-derived HTML channels. Applying them to
                    # synthetic JSON joins creates matches that no scalar or
                    # container exposes.
                    semantic_projection=(
                        suffix == ".html" and projection_index > 0
                        and not renderer_output
                    ),
                )
        raw_body_hit = False
        literal_projection_hit = False
        if not trusted_vendor and raw_bodies:
            for projection in (
                [] if trusted_projection and suffix == ".json"
                else raw_body_projections
            ):
                for candidate in _canonical_variants(projection):
                    if _raw_body_contains(
                        raw_prefixes, raw_prefix_width, candidate,
                        publication_match_budget, raw_body_file_budget,
                    ):
                        raw_body_hit = True
                        break
                if raw_body_hit:
                    break
        if not trusted_vendor and literal_matcher is not None:
            for projection in (
                [] if trusted_projection and suffix == ".json"
                else raw_body_projections
            ):
                if any(
                    _literal_automaton_contains(
                        literal_matcher, candidate,
                        publication_match_budget, literal_file_budget,
                    )
                    for candidate in _canonical_variants(projection)
                ):
                    literal_projection_hit = True
                    break
        if raw_body_hit or ordered_hit:
            if raw_bodies:
                errors.append(location + ":raw-source-body")
        # An ordered match may be a raw body or a structured literal.  Both
        # rule IDs are intentionally opaque and fail closed; reporting the
        # literal class as well avoids a second attacker-controlled pass solely
        # to distinguish two equally blocking source-derived sets.
        if (
            literal_projection_hit or ordered_hit
        ) and ordered_literal_values:
            errors.append(location + ":private-source-literal")
    if errors:
        raise ValueError("Artifact privacy check failed: " + "; ".join(errors[:20]) +
                         "; total=" + str(len(errors)))
    return {"privacy": "PASS", "textual_files_scanned": scanned,
            "sensitive_literal_rules": len(deny),
            "raw_source_hash_rules": len(raw_hashes), "source_downloads": 0}
