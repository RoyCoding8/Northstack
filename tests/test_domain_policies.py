"""Property-based tests for the security-policy domain modules.

Covers src/northstack/domain/url_policy.py (is_loopback_host,
validate_provider_url) and src/northstack/domain/secrets_policy.py
(is_sensitive_env_name, validate_env_allowlist).
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from urllib.parse import urlunparse

from northstack.domain.secrets_policy import (
    is_sensitive_env_name,
    validate_env_allowlist,
)
from northstack.domain.url_policy import (
    is_loopback_host,
    validate_provider_url,
)

_HOST_ALPHABET = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
_KEYWORDS = (
    "KEY",
    "TOKEN",
    "SECRET",
    "PASSWORD",
    "PASSWD",
    "AUTH",
    "CREDENTIAL",
)


def _wrap(host: str) -> str:
    return f"[{host}]" if ":" in host else host


def _url(scheme, host, *, userinfo=None, query="", fragment="", path=""):
    netloc = ""
    if userinfo is not None:
        netloc = f"{userinfo}@"
    netloc += host
    return urlunparse((scheme, netloc, path, "", query, fragment))


def _localhost_variant():
    cases = [st.sampled_from([c.upper(), c.lower()]) for c in "localhost"]
    return st.builds(lambda *cs: "".join(cs), *cases)


domain_host = st.text(alphabet=_HOST_ALPHABET, min_size=1, max_size=30)
nonloop_domain = domain_host.filter(lambda s: s.lower() != "localhost")
ipv4_bare = st.builds(str, st.ip_addresses().filter(lambda ip: ip.version == 4))
ipv6_bare = st.builds(str, st.ip_addresses().filter(lambda ip: ip.version == 6))

general_host_bare = st.one_of(domain_host, ipv4_bare, ipv6_bare)

nonloop_host_bare = st.one_of(
    nonloop_domain,
    st.builds(str, st.ip_addresses().filter(lambda ip: ip.version == 4 and not ip.is_loopback)),
    st.builds(str, st.ip_addresses().filter(lambda ip: ip.version == 6 and not ip.is_loopback)),
)

loopback_host_bare = st.one_of(
    _localhost_variant(),
    st.builds(
        lambda a, b, c: f"127.{a}.{b}.{c}",
        st.integers(0, 255),
        st.integers(0, 255),
        st.integers(0, 255),
    ),
    st.just("::1"),
)


@given(
    scheme=st.one_of(
        st.sampled_from(["ftp", "file", "ws", "wss", "gopher", "ssh", "mailto", "custom"]),
        st.text(alphabet=_HOST_ALPHABET, min_size=2, max_size=12).filter(
            lambda s: s not in ("http", "https")
        ),
    ),
    host=domain_host,
)
@settings(max_examples=200)
def test_u1_non_http_scheme_raises(scheme, host):
    """U1: any non-http/https scheme raises ValueError."""
    with pytest.raises(ValueError):
        validate_provider_url(_url(scheme, _wrap(host)), credentialed=False)


@given(host=domain_host)
@settings(max_examples=200)
def test_u2_userinfo_raises(host):
    """U2: a URL containing userinfo raises ValueError."""
    with pytest.raises(ValueError):
        validate_provider_url(_url("https", _wrap(host), userinfo="user:pass"), credentialed=False)


@given(host=domain_host)
@settings(max_examples=200)
def test_u3_query_raises(host):
    """U3: a non-empty query string raises ValueError."""
    with pytest.raises(ValueError):
        validate_provider_url(_url("https", _wrap(host), query="x=1"), credentialed=False)


@given(host=domain_host)
@settings(max_examples=200)
def test_u4_fragment_raises(host):
    """U4: a non-empty fragment raises ValueError."""
    with pytest.raises(ValueError):
        validate_provider_url(_url("https", _wrap(host), fragment="frag"), credentialed=False)


@given(scheme=st.sampled_from(["http", "https"]))
@settings(max_examples=200)
def test_u5_no_hostname_raises(scheme):
    """U5: a URL with no hostname raises ValueError."""
    with pytest.raises(ValueError):
        validate_provider_url(urlunparse((scheme, "", "", "", "", "")), credentialed=False)


@given(host=nonloop_host_bare)
@settings(max_examples=200)
def test_u6_credentialed_http_nonloopback_raises(host):
    """U6: credentialed http to a non-loopback host (insecure off) raises."""
    with pytest.raises(ValueError):
        validate_provider_url(
            _url("http", _wrap(host)),
            credentialed=True,
            allow_insecure_http=False,
        )


@given(host=loopback_host_bare)
@settings(max_examples=200)
def test_u7_credentialed_http_loopback_ok(host):
    """U7: credentialed http to a loopback host does not raise."""
    validate_provider_url(_url("http", _wrap(host)), credentialed=True, allow_insecure_http=False)


@given(host=nonloop_host_bare)
@settings(max_examples=200)
def test_u8_credentialed_http_nonloopback_insecure_ok(host):
    """U8: credentialed http non-loopback with allow_insecure_http does not raise."""
    validate_provider_url(_url("http", _wrap(host)), credentialed=True, allow_insecure_http=True)


@given(host=general_host_bare, credentialed=st.booleans())
@settings(max_examples=200)
def test_u9_https_any_host_ok(host, credentialed):
    """U9: https with a valid host and no userinfo/query/fragment never raises."""
    validate_provider_url(_url("https", _wrap(host)), credentialed=credentialed)


@given(
    host=st.one_of(
        _localhost_variant(),
        st.builds(
            lambda a, b, c: f"127.{a}.{b}.{c}",
            st.integers(0, 255),
            st.integers(0, 255),
            st.integers(0, 255),
        ),
        st.just("::1"),
    )
)
@settings(max_examples=200)
def test_u10_loopback_true(host):
    """U10: is_loopback_host is True for localhost / 127.x / ::1."""
    assert is_loopback_host(host) is True


@given(
    host=st.one_of(
        st.none(),
        st.builds(str, st.ip_addresses().filter(lambda ip: ip.version == 4 and not ip.is_loopback)),
        st.builds(str, st.ip_addresses().filter(lambda ip: ip.version == 6 and not ip.is_loopback)),
        nonloop_domain,
    )
)
@settings(max_examples=200)
def test_u10_loopback_false(host):
    """U10: is_loopback_host is False for None, public IPs, non-address strings."""
    assert is_loopback_host(host) is False


@given(name=st.text(alphabet=_HOST_ALPHABET + "_", min_size=0, max_size=30))
@settings(max_examples=200)
def test_s1_case_insensitive(name):
    """S1: is_sensitive_env_name is case-insensitive."""
    base = is_sensitive_env_name(name)
    assert base == is_sensitive_env_name(name.upper())
    assert base == is_sensitive_env_name(name.lower())


@given(
    keyword=st.sampled_from(_KEYWORDS),
    token=st.text(alphabet=_HOST_ALPHABET, min_size=1, max_size=15),
)
@settings(max_examples=200)
def test_s2_keyword_tokens_sensitive(keyword, token):
    """S2: P_K, K_P, and P_K_P are all sensitive for any keyword and ASCII token."""
    assert is_sensitive_env_name(f"{token}_{keyword}") is True
    assert is_sensitive_env_name(f"{keyword}_{token}") is True
    assert is_sensitive_env_name(f"{token}_{keyword}_{token}") is True


@given(
    names=st.lists(
        st.text(alphabet=_HOST_ALPHABET + "_", min_size=0, max_size=20),
        max_size=20,
    )
)
@settings(max_examples=200)
def test_s3_allowlist_raises_iff_sensitive(names):
    """S3: validate_env_allowlist raises iff at least one name is sensitive."""
    expected = any(is_sensitive_env_name(n) for n in names)
    if expected:
        with pytest.raises(ValueError):
            validate_env_allowlist(names)
    else:
        validate_env_allowlist(names)


@given(
    names=st.lists(
        st.text(alphabet=_HOST_ALPHABET + "_", min_size=1, max_size=20),
        max_size=20,
    ).filter(lambda ns: not any(is_sensitive_env_name(n) for n in ns))
)
@settings(max_examples=200)
def test_s4_returns_input_on_success(names):
    """S4: on success validate_env_allowlist returns a list equal to its input."""
    assert validate_env_allowlist(names) == names


@pytest.mark.parametrize(
    "name",
    [
        "API_KEY",
        "OPENAI_API_KEY",
        "AWS_SECRET_ACCESS_KEY",
        "GITHUB_TOKEN",
        "DB_PASSWORD",
        "HTTP_AUTH",
        "MY_CREDENTIAL",
        "apikey",
        "Bearer_Token",
        "PASSWD",
    ],
)
def test_s5_corpus_sensitive(name):
    """S5: the required corpus names are all classified sensitive."""
    assert is_sensitive_env_name(name) is True


@pytest.mark.parametrize(
    "name",
    ["AWS_KEYS", "TOKENS", "SECRETS", "API_KEYS", "PASSWORDS", "KEYS", "CREDENTIALS"],
)
def test_s6_plural_forms_sensitive(name):
    """S6: plural forms are sensitive (docstring promises fail-safe over-reject)."""
    assert is_sensitive_env_name(name) is True


def test_s7_plural_rejected_by_allowlist():
    """S7: the allowlist gate, not just the predicate, refuses a plural secret name."""
    with pytest.raises(ValueError, match="AWS_KEYS"):
        validate_env_allowlist(["PATH", "AWS_KEYS", "HOME"])
    assert validate_env_allowlist(["PATH", "HOME", "LANG"]) == ["PATH", "HOME", "LANG"]
