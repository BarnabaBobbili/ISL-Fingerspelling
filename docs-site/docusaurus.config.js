// @ts-check
import {themes as prismThemes} from 'prism-react-renderer';

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'ISL Fingerspelling',
  tagline: 'Recognition and 3D synthesis of Indian Sign Language fingerspelling',
  favicon: 'img/favicon.svg',

  // The site is built to be read locally (npm start) or hosted anywhere static.
  url: 'http://localhost',
  baseUrl: '/',
  organizationName: 'BarnabaBobbili',
  projectName: 'ISL-Fingerspelling',

  onBrokenLinks: 'throw',
  markdown: {mermaid: true, hooks: {onBrokenMarkdownLinks: 'throw'}},
  themes: ['@docusaurus/theme-mermaid'],

  i18n: {defaultLocale: 'en', locales: ['en']},

  presets: [
    [
      'classic',
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: {
          routeBasePath: '/',           // documentation-only site: docs live at the root
          sidebarPath: './sidebars.js',
        },
        blog: false,
        theme: {customCss: './src/css/custom.css'},
      }),
    ],
  ],

  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      colorMode: {defaultMode: 'dark', respectPrefersColorScheme: true},
      navbar: {
        title: 'ISL Fingerspelling',
        logo: {alt: 'ISL Fingerspelling', src: 'img/logo.svg'},
        items: [
          {type: 'docSidebar', sidebarId: 'manual', position: 'left', label: 'Manual'},
          {href: 'https://github.com/BarnabaBobbili/ISL-Fingerspelling', label: 'GitHub', position: 'right'},
        ],
      },
      footer: {
        style: 'dark',
        copyright: `ISL Fingerspelling — M.Tech technical enrichment project. Built with Docusaurus.`,
      },
      prism: {
        theme: prismThemes.github,
        darkTheme: prismThemes.dracula,
        additionalLanguages: ['bash', 'python', 'json', 'powershell'],
      },
      mermaid: {theme: {light: 'neutral', dark: 'dark'}},
    }),
};

export default config;
