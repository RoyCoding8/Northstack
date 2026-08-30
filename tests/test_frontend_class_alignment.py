"""Frontend class-alignment gate: every class token the JS emits must have a
CSS definition.

The northstack control surface is a vanilla-JS + hand-rolled MD3 CSS app.
Its single most dangerous failure mode is drift between the two layers: JS
builds elements with ``class: "foo bar"`` while the CSS styles *different*
names, so the markup renders unstyled (the original "Dashboard mixes with
this that" bug was exactly this -- the app shell grid keyed on ``#app`` but JS
built ``<div class="app">``, disabling the grid).

This test is the automated mirror of that audit. It parses every CSS class
selector out of ``components.css`` and ``app.css``, parses every class token
out of the JS ``class: "..."`` / ``classList.*`` literals, and fails if any
JS-emitted class has no matching CSS rule.

Allowances for classes that are *composed* at runtime rather than written
literally (e.g. ``budget-bar__fill--ok`` is built as a ``"...--" + x`` suffix
in JS) are listed in ``DYNAMIC_OK`` -- those literal *base* names are still
checked against CSS, only the dynamic suffixes are exempt.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import get_args

import pytest

from northstack.config import ModelProfile, Protocol

STATIC = (
    Path(__file__).resolve().parents[1] / "src" / "northstack" / "interfaces" / "web" / "static"
)
CSS_FILES = ["components.css", "app.css"]

# Class tokens JS builds at runtime by string concatenation (e.g.
# ``"stepper__step--" + x``). The literal *base* before the suffix is still
# verified against CSS; only these fully-dynamic compound forms are exempt.
DYNAMIC_OK: set[str] = {
    "stepper__step--done",
    "stepper__step--current",
    "stepper__step--terminal",
    "budget-bar__fill--ok",
    "budget-bar__fill--warn",
    "budget-bar__fill--over",
    "badge--dot",  # the dot variant; the ok/none/unset suffix is dynamic
    "gnode--",  # graph node status suffix is dynamic
}

# Pseudo-class suffixes / dot-variant markers that appear in class strings but
# are not standalone CSS classes (state flags built dynamically).
_OK_SUFFIX_WORDS = {"ok", "none", "unset", "verified", "failed", "active", "dirty"}

# Match a CSS class selector token: a leading dot followed by an identifier.
_CSS_CLASS_RE = re.compile(r"\.([A-Za-z_][\w-]*)")

# Match JS class-attribute string literals: class: "..." or class:'...' or
# class:`...`, plus explicit classList.add/toggle/remove/contains("...").
_JS_CLASS_RE = re.compile(r"""class:\s*["'`]([^"'`]+)["'`]""")
_JS_CL_RE = re.compile(r"""classList\.(?:add|toggle|remove|contains)\(\s*["'`]([^"'`]+)["'`]""")


def _css_classes() -> set[str]:
    found: set[str] = set()
    for name in CSS_FILES:
        p = STATIC / name
        if not p.is_file():
            continue
        text = p.read_text(encoding="utf-8")
        # strip block comments so commented-out rules don't count as definitions
        text = re.sub(r"/\*[\s\S]*?\*/", "", text)
        found.update(_CSS_CLASS_RE.findall(text))
    return found


def _js_class_tokens() -> set[str]:
    js_dir = STATIC / "js"
    found: set[str] = set()
    for p in sorted(js_dir.rglob("*.js")):
        text = p.read_text(encoding="utf-8")
        for m in _JS_CLASS_RE.finditer(text):
            for tok in m.group(1).split():
                tok = tok.strip()
                if tok and not tok.startswith("${"):
                    found.add(tok)
        for m in _JS_CL_RE.finditer(text):
            tok = m.group(1).strip()
            if tok and not tok.startswith("${"):
                found.add(tok)
    return found


def test_every_js_class_has_a_css_definition() -> None:
    """No JS-emitted class token may lack a CSS rule (the 'shell mixes' bug)."""
    css = _css_classes()
    js = _js_class_tokens()
    assert css, "no CSS class selectors parsed -- CSS_FILES path wrong?"

    dangling = sorted(
        c for c in js if c not in css and c not in DYNAMIC_OK and c not in _OK_SUFFIX_WORDS
    )
    if dangling:
        pytest.fail(
            "JS emits class token(s) with no matching CSS definition. Either "
            "add the CSS rule or align the JS to a defined name:\n  " + ", ".join(dangling)
        )


def test_app_shell_grid_root_is_id_app() -> None:
    """The shell grid must key on #app (app.css) AND JS must build #app.

    Regression guard for the root cause of the original 'dashboard mixes'
    layout collapse: CSS laid out on ``#app`` but JS built ``class="app"``.
    """
    app_css = (STATIC / "app.css").read_text(encoding="utf-8")
    assert re.search(r"#app\s*\{", app_css), "app.css must define the #app grid root"

    app_js = (STATIC / "js" / "app.js").read_text(encoding="utf-8")
    # JS must build the root with id="app", not class="app".
    assert 'id: "app"' in app_js or 'id:"app"' in app_js, (
        'app.js buildShell() must create the root with id="app" '
        "(the CSS grid keys on #app, not .app)"
    )
    assert not re.search(r'class:\s*"app"', app_js), (
        'app.js must not set class="app" on the shell root '
        "(that disabled the #app grid in the original bug)"
    )


def test_drawer_state_on_app_element() -> None:
    """The mobile drawer must open via #app[data-drawer], not a #drawer element.

    CSS opens the off-canvas rail with ``#app[data-drawer="open"] .navrail``.
    JS must set that attribute on #app; toggling ``data-open`` on a stray
    ``#drawer`` element (which CSS never reads) was the original dead-drawer bug.
    """
    app_css = (STATIC / "app.css").read_text(encoding="utf-8")
    assert 'data-drawer="open"' in app_css, (
        'app.css must open the rail via #app[data-drawer="open"]'
    )
    app_js = (STATIC / "js" / "app.js").read_text(encoding="utf-8")
    assert '"data-drawer", "open"' in app_js or "'data-drawer','open'" in app_js, (
        "app.js must open the drawer by setting data-drawer=open on #app"
    )


def test_self_hosted_fonts_have_no_cdn_dependency() -> None:
    """Fira must load from /static/fonts, never a network CDN (offline-first)."""
    tokens = (STATIC / "tokens.css").read_text(encoding="utf-8")
    # @font-face src must point at /static/fonts/*.woff2, not https://
    for m in re.finditer(r"@font-face\s*\{[^}]*\}", tokens, re.DOTALL):
        block = m.group(0)
        assert "https://" not in block and "http://" not in block, (
            "self-hosted @font-face must not reference a remote URL"
        )
        assert "/static/fonts/" in block, "@font-face src must point at /static/fonts/*.woff2"
    fonts_dir = STATIC / "fonts"
    assert fonts_dir.is_dir(), f"self-hosted fonts dir missing: {fonts_dir}"
    woff2 = list(fonts_dir.glob("*.woff2"))
    assert woff2, "no .woff2 font files in static/fonts/"


def test_profile_editor_uses_backend_enum_values() -> None:
    """Profile controls must not emit values rejected by Protocol/Capability."""
    profiles_js = (STATIC / "js" / "views" / "profiles.js").read_text(encoding="utf-8")

    assert 'const CAPS = ["tool_use", "native_json_schema", "vision", "streaming"]' in profiles_js
    protos = ", ".join(f'"{p.value}"' for p in Protocol)
    assert f"const PROTOS = [{protos}]" in profiles_js
    assert "allow_insecure_http: p.allow_insecure_http === true" in profiles_js
    params = ", ".join(
        f'"{v}"' for v in get_args(ModelProfile.model_fields["token_limit_param"].annotation)
    )
    assert f"const TOKEN_LIMIT_PARAMS = [{params}]" in profiles_js
    for invalid in ('"read"', '"write"', '"create"', '"replace"', '"openrouter_chat"'):
        assert invalid not in profiles_js


def test_routing_ui_omits_empty_route_mappings() -> None:
    """RouteMapping requires at least one profile; empty cards mean no mapping."""
    routing_js = (STATIC / "js" / "views" / "routing.js").read_text(encoding="utf-8")

    assert ".filter(entry => entry.profiles.length > 0)" in routing_js
    assert "disabled: !p.available" in routing_js


def _z_index(selector: str) -> int:
    """The z-index a class carries across the built CSS bundles."""
    for name in CSS_FILES:
        css = (STATIC / name).read_text(encoding="utf-8")
        for m in re.finditer(re.escape("." + selector) + r"\s*\{([^}]*)\}", css):
            z = re.search(r"z-index:\s*(-?\d+)", m.group(1))
            if z:
                return int(z.group(1))
    raise AssertionError(f"no z-index found for .{selector}")


def test_the_select_menu_stacks_above_every_overlay_it_opens_inside():
    """selectMenu appends its popup to <body>, so CSS layering is the only
    thing keeping it in front of the scrim it was opened over. Below the
    scrim the popup is invisible AND the click lands on the scrim, closing
    the dialog instead of picking an option."""
    popup = _z_index("select-menu__list")

    for over in ("scrim", "palette", "drawer-scrim"):
        assert popup > _z_index(over), f".select-menu__list must stack above .{over}"


def test_dialogs_that_host_a_select_menu_do_not_trap_its_popup():
    """The popup escapes the dialog by living on <body>; a dialog-scoped
    append would be clipped by .dialog's overflow:auto."""
    util = (STATIC / "js" / "util.js").read_text(encoding="utf-8")

    assert "document.body.appendChild(list)" in util
    # Capture phase: a scrim click must dismiss the menu without reaching the
    # scrim's own bubble handler and closing the dialog behind it.
    assert 'document.addEventListener("click", onOutside, true)' in util
