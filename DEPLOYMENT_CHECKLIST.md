# Deployment Checklist for GitHub Pages

## ✅ Changes Made

1. **Updated `astro.config.mjs`**:
   - Set `site: 'https://ajinkyakulkarni.github.io'`
   - Set `base: '/folio'`

2. **Fixed asset paths**:
   - Updated favicon path in BaseLayout to use `import.meta.env.BASE_URL`
   - Updated GitHub link to your repository

3. **Created GitHub Actions workflow** (`.github/workflows/deploy.yml`)

4. **Created `.gitignore`** file

## 📋 Steps to Deploy the Fix

1. **Commit and push these changes**:
   ```bash
   git add .
   git commit -m "Fix CSS and asset paths for GitHub Pages deployment"
   git push
   ```

2. **Wait for GitHub Actions** to rebuild and deploy (check Actions tab)

3. **Clear browser cache** and visit: https://ajinkyakulkarni.github.io/folio/

## 🔍 If CSS Still Appears Broken

1. **Check browser console** for any 404 errors
2. **Verify the build output** in GitHub Actions logs
3. **Try hard refresh**: Ctrl+Shift+R (Windows/Linux) or Cmd+Shift+R (Mac)

## 📝 Notes

- The site uses Tailwind CSS v4 which is compiled during build
- All internal links should work correctly with the base path
- The favicon should now load properly
- GitHub repository link has been updated

## 🚀 Future Updates

When you make changes:
1. Make your edits locally
2. Run `npm run build` to test
3. Commit and push to main branch
4. GitHub Actions will automatically deploy