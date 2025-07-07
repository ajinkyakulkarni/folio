const fs = require('fs');
const path = require('path');
const yaml = require('js-yaml');
const matter = require('gray-matter');

// Load review rules
const rulesPath = path.join(process.cwd(), 'review-rules.yml');
const rules = yaml.load(fs.readFileSync(rulesPath, 'utf8'));

// Parse command line arguments
const changedFiles = process.argv[2] ? process.argv[2].split(' ') : [];

// Initialize review results
const reviewResults = {
  approved: true,
  summary: '',
  issues: [],
  suggestions: []
};

// Simple LLM review simulation
// In production, replace this with actual API calls to OpenAI/Anthropic
async function performLLMReview(content, frontmatter, contentType) {
  const issues = [];
  const suggestions = [];
  
  // Simulate content quality check
  const wordCount = content.split(/\s+/).length;
  
  // Check for low-quality indicators
  const lowQualityPatterns = [
    { pattern: /lorem ipsum/i, message: 'Placeholder text detected' },
    { pattern: /test test test/i, message: 'Test content detected' },
    { pattern: /asdf|qwerty/i, message: 'Keyboard mashing detected' },
    { pattern: /(.)\1{10,}/i, message: 'Repeated characters detected' }
  ];
  
  lowQualityPatterns.forEach(({ pattern, message }) => {
    if (pattern.test(content)) {
      issues.push({
        type: 'quality',
        message: message
      });
    }
  });
  
  // Check title quality
  if (frontmatter.title) {
    // Check for clickbait patterns
    const clickbaitPatterns = [
      /you won't believe/i,
      /this one trick/i,
      /doctors hate/i,
      /number \d+ will shock you/i
    ];
    
    clickbaitPatterns.forEach(pattern => {
      if (pattern.test(frontmatter.title)) {
        issues.push({
          type: 'title',
          message: 'Title appears to be clickbait'
        });
      }
    });
  }
  
  // Provide suggestions based on content analysis
  if (contentType === 'blog') {
    // Check for missing sections
    if (!content.includes('## Introduction') && !content.includes('## Overview')) {
      suggestions.push('Consider adding an introduction section to provide context');
    }
    
    if (!content.includes('## Conclusion') && !content.includes('## Summary')) {
      suggestions.push('Consider adding a conclusion to summarize key points');
    }
    
    // Check for code blocks in technical posts
    const technicalTags = ['javascript', 'python', 'nodejs', 'react', 'coding'];
    const hasTechnicalTag = frontmatter.tags?.some(tag => 
      technicalTags.some(tech => tag.toLowerCase().includes(tech))
    );
    
    if (hasTechnicalTag && !content.includes('```')) {
      suggestions.push('Technical posts benefit from code examples. Consider adding code snippets');
    }
    
    // Check for proper heading structure
    const headings = content.match(/^#{1,6} .+$/gm) || [];
    if (headings.length < 3 && wordCount > 500) {
      suggestions.push('Long posts benefit from clear section headings for better readability');
    }
  }
  
  // Simulate readability check
  const avgSentenceLength = content.split(/[.!?]+/).length > 0 
    ? wordCount / content.split(/[.!?]+/).length 
    : 0;
    
  if (avgSentenceLength > 25) {
    suggestions.push('Some sentences appear to be quite long. Consider breaking them up for better readability');
  }
  
  return { issues, suggestions };
}

// Main review process
async function reviewContent() {
  let allIssues = [];
  let allSuggestions = [];
  let summaryPoints = [];
  
  for (const filePath of changedFiles) {
    if (!filePath.endsWith('.md') && !filePath.endsWith('.mdx')) {
      continue;
    }
    
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
      
      // Perform LLM review
      const { issues, suggestions } = await performLLMReview(body, frontmatter, contentType);
      
      allIssues.push(...issues);
      allSuggestions.push(...suggestions);
      
      // Add to summary
      if (contentType === 'blog') {
        summaryPoints.push(`New blog post: "${frontmatter.title}" by ${frontmatter.author}`);
      } else if (contentType === 'authors') {
        summaryPoints.push(`New author profile: ${frontmatter.name}`);
      }
      
    } catch (error) {
      allIssues.push({
        type: 'error',
        message: `Failed to review ${filePath}: ${error.message}`
      });
    }
  }
  
  // Compile final results
  reviewResults.approved = allIssues.length === 0;
  reviewResults.issues = allIssues;
  reviewResults.suggestions = [...new Set(allSuggestions)]; // Remove duplicates
  reviewResults.summary = summaryPoints.join('. ') || 'Content review completed.';
  
  // Add feedback message
  if (reviewResults.approved) {
    reviewResults.summary += ' All content meets our quality standards.';
  } else {
    reviewResults.summary += ' Some issues need to be addressed before approval.';
  }
  
  // Output results
  console.log(JSON.stringify(reviewResults, null, 2));
}

// Run the review
reviewContent().catch(error => {
  console.error('Review failed:', error);
  process.exit(1);
});