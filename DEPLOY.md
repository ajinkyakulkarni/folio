# Deploying Folio to GitHub Pages

Follow these steps to deploy your Folio blog to GitHub Pages:

## Prerequisites

1. A GitHub account
2. Git installed on your local machine
3. Node.js and npm installed

## Step-by-Step Deployment Guide

### 1. Update Astro Configuration

Edit `astro.config.mjs` and replace the placeholder values:

```javascript
export default defineConfig({
  site: 'https://YOUR-USERNAME.github.io',  // Replace YOUR-USERNAME
  base: '/YOUR-REPO-NAME',                  // Replace YOUR-REPO-NAME
  // ... rest of config
});
```

Example:
- If your GitHub username is `johndoe` and your repository is `my-folio-blog`
- Set `site: 'https://johndoe.github.io'`
- Set `base: '/my-folio-blog'`

### 2. Create a GitHub Repository

1. Go to [GitHub](https://github.com) and create a new repository
2. Name it something like `folio-blog` or `my-blog`
3. Keep it public (required for free GitHub Pages hosting)
4. Don't initialize with README, .gitignore, or license

### 3. Initialize Git and Push Code

Run these commands in your project directory:

```bash
# Initialize git repository
git init

# Add all files
git add .

# Create initial commit
git commit -m "Initial commit"

# Add your GitHub repository as origin
git remote add origin https://github.com/YOUR-USERNAME/YOUR-REPO-NAME.git

# Push to main branch
git push -u origin main
```

### 4. Enable GitHub Pages

1. Go to your repository on GitHub
2. Click on **Settings** tab
3. Scroll down to **Pages** section in the left sidebar
4. Under **Source**, select **GitHub Actions**

### 5. Deploy

The deployment will happen automatically when you push to the main branch. The GitHub Actions workflow we've set up will:

1. Build your Astro site
2. Upload it as an artifact
3. Deploy it to GitHub Pages

You can monitor the deployment:
1. Go to the **Actions** tab in your repository
2. Watch the workflow run
3. Once complete (green checkmark), your site will be live!

### 6. Access Your Site

Your blog will be available at:
```
https://YOUR-USERNAME.github.io/YOUR-REPO-NAME/
```

## Updating Your Blog

To update your blog with new content:

```bash
# Make your changes
git add .
git commit -m "Add new blog post"
git push
```

GitHub Actions will automatically rebuild and deploy your site.

## Custom Domain (Optional)

To use a custom domain:

1. Add a `CNAME` file in the `public/` directory with your domain
2. Update `astro.config.mjs`:
   ```javascript
   site: 'https://yourdomain.com',
   base: '/',  // Remove the base path for custom domains
   ```
3. Configure your domain's DNS settings to point to GitHub Pages

## Troubleshooting

### Build Fails
- Check the Actions tab for error messages
- Ensure all dependencies are in `package.json`
- Run `npm run build` locally to test

### 404 Errors
- Verify the `base` path in `astro.config.mjs` matches your repository name
- Ensure the site has finished deploying (check Actions tab)

### Images/Assets Not Loading
- Make sure all asset paths start with `/` or use Astro's assets handling
- Check that the `base` path is correctly configured

## Additional Notes

- The first deployment might take a few minutes to become available
- GitHub Pages has a soft limit of 10 builds per hour
- The site must be under 1GB in size
- Always test your build locally with `npm run build` before pushing