/*
  Tailwind config for the northstack control surface.

  Engine: Material Design 3 via @material-tailwind/html (the vanilla/HTML build
  of Material Tailwind), layered on Tailwind v3. `withMT()` injects the MD3
  component utilities (e.g. the class strings that style `.btn`-equivalents,
  ripples via data-ripple-light) and the MD3 default theme shape.

  COLOR DISCIPLINE: no hex lives anywhere except tokens.css. So every Tailwind
  color token below resolves to a var(--md-sys-color-*) / var(--p-*) from
  tokens.css.  Built utilities therefore emit var(...) references, never raw
  hex -- keeping the token-compliance gate's intent (single hex source).
*/
const withMT = require("@material-tailwind/html/utils/withMT");

module.exports = withMT({
  content: [
    "./src/northstack/interfaces/web/static/index.html",
    "./src/northstack/interfaces/web/static/js/**/*.js",
    "./tailwind-src.css",
  ],
  theme: {
    extend: {
      colors: {
        // Semantic MD3 colors -> driven by [data-theme] in tokens.css.
        primary: "var(--md-sys-color-primary)",
        "on-primary": "var(--md-sys-color-on-primary)",
        "primary-container": "var(--md-sys-color-primary-container)",
        "on-primary-container": "var(--md-sys-color-on-primary-container)",
        secondary: "var(--md-sys-color-secondary)",
        "on-secondary": "var(--md-sys-color-on-secondary)",
        "secondary-container": "var(--md-sys-color-secondary-container)",
        "on-secondary-container": "var(--md-sys-color-on-secondary-container)",
        tertiary: "var(--md-sys-color-tertiary)",
        "on-tertiary": "var(--md-sys-color-on-tertiary)",
        "tertiary-container": "var(--md-sys-color-tertiary-container)",
        "on-tertiary-container": "var(--md-sys-color-on-tertiary-container)",
        error: "var(--md-sys-color-error)",
        "on-error": "var(--md-sys-color-on-error)",
        "error-container": "var(--md-sys-color-error-container)",
        "on-error-container": "var(--md-sys-color-on-error-container)",
        background: "var(--md-sys-color-background)",
        "on-background": "var(--md-sys-color-on-background)",
        surface: "var(--md-sys-color-surface)",
        "on-surface": "var(--md-sys-color-on-surface)",
        "surface-variant": "var(--md-sys-color-surface-variant)",
        "on-surface-variant": "var(--md-sys-color-on-surface-variant)",
        outline: "var(--md-sys-color-outline)",
        "outline-variant": "var(--md-sys-color-outline-variant)",
      },
      borderRadius: {
        xs: "var(--p-shape-xs)",
        sm: "var(--p-shape-sm)",
        md: "var(--p-shape-md)",
        lg: "var(--p-shape-lg)",
        xl: "var(--p-shape-xl)",
      },
      boxShadow: {
        "elev-1": "var(--p-elev-1)",
        "elev-2": "var(--p-elev-2)",
        "elev-3": "var(--p-elev-3)",
      },
      fontFamily: {
        sans: "var(--p-font-ui)",
        mono: "var(--p-font-mono)",
        symbols: "var(--p-font-symbols)",
      },
      transitionTimingFunction: {
        emph: "var(--p-ease-emph)",
        std: "var(--p-ease-std)",
      },
      // Tailwind's preflight-vars block (*,:after,:before) hard-codes these
      // theme DEFAULTs into every build. Override to tokens so the emitted
      // bundle contains zero raw hex / zero raw rgba — keeping the
      // token-compliance gate (no hex outside tokens.css) intact.
      ringOffsetColor: {
        DEFAULT: "var(--md-sys-color-surface)",
      },
      ringColor: {
        DEFAULT: "var(--md-sys-color-outline)",
      },
    },
  },
  // Preserve existing class names; we @apply MD3 utilities onto them in
  // tailwind-src.css. No @tailwind base reset cascade-fight (tokens.css owns
  // body/html), so layer base accordingly.
  corePlugins: { preflight: false },
  plugins: [],
});
