# Folio Platform

A modern static website platform that combines blogging capabilities with professional portfolios, built with Astro, Tailwind CSS, and automated GitHub-based content management.

## Features

- **Multi-Author Blogging**: Support for multiple authors with individual profiles
- **Professional Portfolios**: Showcase projects, skills, and achievements
- **Featured Content**: Highlight important articles on the homepage
- **Search Functionality**: Find articles and authors easily
- **Taxonomy System**: Organize content with categories and tags
- **GitHub-Based CMS**: Contribute content via pull requests
- **Automated Content Review**: LLM-powered PR reviews for quality control
- **Static Site Generation**: Fast, secure, and SEO-friendly

## Getting Started

### Prerequisites

- Node.js 18 or higher
- npm or yarn
- Git

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/folio.git
cd folio
```

2. Install dependencies:
```bash
npm install
```

3. Start the development server:
```bash
npm run dev
```

4. Open http://localhost:4321 in your browser

### Building for Production

```bash
npm run build
npm run preview
```

## Contributing Content

### Adding a New Blog Post

1. Fork this repository
2. Create a new markdown file in `src/content/blog/`:
```markdown
---
title: "Your Amazing Article Title"
description: "A brief description of your article"
author: your-author-slug
publishDate: 2024-03-20
category: "Web Development"
tags: ["javascript", "tutorial", "webdev"]
heroImage: "https://example.com/image.jpg"
featured: false
draft: false
---

Your article content here...
```

3. Submit a pull request
4. Wait for automated review and approval

### Creating an Author Profile

1. Create a new markdown file in `src/content/authors/`:
```markdown
---
name: "Your Name"
email: "your.email@example.com"
bio: "A brief bio about yourself"
skills: ["JavaScript", "React", "Node.js"]
joinedDate: 2024-03-20
---

Extended bio and information...
```

2. Submit a pull request

## Content Guidelines

### Blog Posts
- Minimum 300 words
- Include at least 2 tags
- Use proper markdown formatting
- Add code examples for technical posts
- Include introduction and conclusion sections

### Author Profiles
- Complete all required fields
- Minimum 100 characters for bio
- Include at least one social link

## Automated Review Process

All content submissions are automatically reviewed for:
- Required fields and formatting
- Content quality and readability
- Prohibited content (spam, inappropriate language)
- Security concerns (exposed secrets)

Review rules are configured in `review-rules.yml`.

## Project Structure

```
folio/
├── src/
│   ├── components/     # Reusable Astro components
│   ├── content/        # Markdown content
│   │   ├── authors/    # Author profiles
│   │   ├── blog/       # Blog posts
│   │   └── portfolio/  # Portfolio items
│   ├── layouts/        # Page layouts
│   ├── pages/          # Route pages
│   └── styles/         # Global styles
├── public/             # Static assets
├── .github/            # GitHub workflows
│   ├── workflows/      # Automated actions
│   └── scripts/        # Review scripts
└── review-rules.yml    # Content review configuration
```

## Commands

| Command                   | Action                                           |
| :------------------------ | :----------------------------------------------- |
| `npm install`             | Installs dependencies                            |
| `npm run dev`             | Starts local dev server at `localhost:4321`      |
| `npm run build`           | Build your production site to `./dist/`          |
| `npm run preview`         | Preview your build locally, before deploying     |

## Deployment

This site can be deployed to any static hosting service:

### Netlify
1. Connect your GitHub repository
2. Build command: `npm run build`
3. Publish directory: `dist`

### Vercel
1. Import your GitHub repository
2. Framework preset: Astro
3. Deploy

### GitHub Pages
1. Update `astro.config.mjs` with your site URL
2. Use GitHub Actions for automated deployment

## Technologies Used

- **Astro** - Static site generator
- **Tailwind CSS** - Utility-first CSS framework
- **TypeScript** - Type safety
- **Markdown/MDX** - Content format
- **GitHub Actions** - CI/CD and content review

## License

MIT License - feel free to use this project for your own platform!

## Support

- Create an issue for bug reports
- Start a discussion for feature requests
- Check the wiki for detailed documentation
