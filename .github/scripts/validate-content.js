const fs = require('fs');
const path = require('path');
const yaml = require('js-yaml');
const matter = require('gray-matter');

// Load review rules
const rulesPath = path.join(process.cwd(), 'review-rules.yml');
const rules = yaml.load(fs.readFileSync(rulesPath, 'utf8'));

// Parse command line arguments
const changedFiles = process.argv[2] ? process.argv[2].split(' ') : [];

let hasErrors = false;
const errors = [];

// Validate each changed file
changedFiles.forEach(filePath => {
  if (!filePath.endsWith('.md') && !filePath.endsWith('.mdx')) {
    return;
  }

  console.log(`Validating ${filePath}...`);

  try {
    const content = fs.readFileSync(filePath, 'utf8');
    const { data: frontmatter, content: body } = matter(content);

    // Determine content type
    let contentType = 'blog';
    if (filePath.includes('/authors/')) {
      contentType = 'authors';
    } else if (filePath.includes('/portfolio/')) {
      contentType = 'portfolio';
    }

    // Check required fields
    const requiredFields = rules.content_rules.required_fields[contentType];
    if (requiredFields) {
      requiredFields.forEach(field => {
        if (!frontmatter[field]) {
          errors.push({
            file: filePath,
            type: 'missing_field',
            message: `Missing required field: ${field}`
          });
          hasErrors = true;
        }
      });
    }

    // Validate word count for blog posts
    if (contentType === 'blog') {
      const wordCount = body.split(/\s+/).length;
      const { min, max } = rules.content_rules.word_count;
      
      if (wordCount < min) {
        errors.push({
          file: filePath,
          type: 'word_count',
          message: `Content too short. Minimum ${min} words required, found ${wordCount}`
        });
        hasErrors = true;
      }
      
      if (wordCount > max) {
        errors.push({
          file: filePath,
          type: 'word_count',
          message: `Content too long. Maximum ${max} words allowed, found ${wordCount}`
        });
        hasErrors = true;
      }

      // Validate title
      if (frontmatter.title) {
        const titleRules = rules.content_rules.guidelines.title;
        if (frontmatter.title.length < titleRules.min_length) {
          errors.push({
            file: filePath,
            type: 'title',
            message: `Title too short. Minimum ${titleRules.min_length} characters required`
          });
          hasErrors = true;
        }
        if (frontmatter.title.length > titleRules.max_length) {
          errors.push({
            file: filePath,
            type: 'title',
            message: `Title too long. Maximum ${titleRules.max_length} characters allowed`
          });
          hasErrors = true;
        }
        if (titleRules.no_all_caps && frontmatter.title === frontmatter.title.toUpperCase()) {
          errors.push({
            file: filePath,
            type: 'title',
            message: 'Title should not be in all caps'
          });
          hasErrors = true;
        }
      }

      // Validate description
      if (frontmatter.description) {
        const descRules = rules.content_rules.guidelines.description;
        if (frontmatter.description.length < descRules.min_length) {
          errors.push({
            file: filePath,
            type: 'description',
            message: `Description too short. Minimum ${descRules.min_length} characters required`
          });
          hasErrors = true;
        }
        if (frontmatter.description.length > descRules.max_length) {
          errors.push({
            file: filePath,
            type: 'description',
            message: `Description too long. Maximum ${descRules.max_length} characters allowed`
          });
          hasErrors = true;
        }
      }

      // Validate tags
      if (frontmatter.tags) {
        const tagRules = rules.content_rules.guidelines.tags;
        if (frontmatter.tags.length < tagRules.min_count) {
          errors.push({
            file: filePath,
            type: 'tags',
            message: `Too few tags. Minimum ${tagRules.min_count} required`
          });
          hasErrors = true;
        }
        if (frontmatter.tags.length > tagRules.max_count) {
          errors.push({
            file: filePath,
            type: 'tags',
            message: `Too many tags. Maximum ${tagRules.max_count} allowed`
          });
          hasErrors = true;
        }
        
        // Validate tag format
        const tagPattern = new RegExp(tagRules.allowed_pattern);
        frontmatter.tags.forEach(tag => {
          if (!tagPattern.test(tag)) {
            errors.push({
              file: filePath,
              type: 'tags',
              message: `Invalid tag format: "${tag}". Tags must match pattern: ${tagRules.allowed_pattern}`
            });
            hasErrors = true;
          }
        });
      }

      // Validate category
      if (frontmatter.category) {
        const allowedCategories = rules.content_rules.guidelines.categories.allowed;
        if (!allowedCategories.includes(frontmatter.category)) {
          errors.push({
            file: filePath,
            type: 'category',
            message: `Invalid category: "${frontmatter.category}". Allowed categories: ${allowedCategories.join(', ')}`
          });
          hasErrors = true;
        }
      }
    }

    // Check for prohibited content
    const prohibitedPatterns = rules.content_rules.prohibited_content.spam_patterns;
    const contentLower = body.toLowerCase();
    
    prohibitedPatterns.forEach(pattern => {
      if (contentLower.includes(pattern.toLowerCase())) {
        errors.push({
          file: filePath,
          type: 'prohibited_content',
          message: `Prohibited content found: "${pattern}"`
        });
        hasErrors = true;
      }
    });

    // Check for security patterns
    const securityPatterns = rules.content_rules.prohibited_content.security_patterns;
    securityPatterns.forEach(pattern => {
      if (body.includes(pattern)) {
        errors.push({
          file: filePath,
          type: 'security',
          message: `Potential security issue found: "${pattern}"`
        });
        hasErrors = true;
      }
    });

  } catch (error) {
    errors.push({
      file: filePath,
      type: 'parse_error',
      message: `Failed to parse file: ${error.message}`
    });
    hasErrors = true;
  }
});

// Output results
if (hasErrors) {
  console.error('\n❌ Validation failed with the following errors:\n');
  errors.forEach(error => {
    console.error(`  ${error.file}:`);
    console.error(`    - [${error.type}] ${error.message}`);
  });
  process.exit(1);
} else {
  console.log('\n✅ All content validation checks passed!');
}