import { defineCollection, z } from 'astro:content';

const authorCollection = defineCollection({
  type: 'content',
  schema: z.object({
    name: z.string(),
    email: z.string().email(),
    avatar: z.string().optional(),
    bio: z.string(),
    location: z.string().optional(),
    company: z.string().optional(),
    position: z.string().optional(),
    website: z.string().url().optional(),
    github: z.string().optional(),
    linkedin: z.string().optional(),
    twitter: z.string().optional(),
    skills: z.array(z.string()),
    expertise: z.array(z.string()),
    education: z.array(z.object({
      degree: z.string(),
      institution: z.string(),
      year: z.number(),
      field: z.string().optional(),
    })).optional(),
    experience: z.array(z.object({
      title: z.string(),
      company: z.string(),
      startDate: z.date(),
      endDate: z.date().optional(),
      description: z.string().optional(),
      skills: z.array(z.string()).optional(),
    })).optional(),
    achievements: z.array(z.string()).optional(),
    videos: z.array(z.object({
      title: z.string(),
      description: z.string(),
      youtubeId: z.string(),
      publishedDate: z.date(),
    })).optional(),
    isActive: z.boolean().default(true),
    joinedDate: z.date(),
  }),
});

const blogCollection = defineCollection({
  type: 'content',
  schema: z.object({
    title: z.string(),
    description: z.string(),
    author: z.string(), // Reference to author slug
    publishDate: z.date(),
    updateDate: z.date().optional(),
    heroImage: z.string().optional(),
    category: z.string(),
    tags: z.array(z.string()),
    featured: z.boolean().default(false),
    draft: z.boolean().default(false),
    readingTime: z.number().optional(), // in minutes
    likes: z.number().default(0),
    views: z.number().default(0),
  }),
});

const portfolioCollection = defineCollection({
  type: 'content',
  schema: z.object({
    title: z.string(),
    description: z.string(),
    author: z.string(), // Reference to author slug
    projectDate: z.date(),
    projectUrl: z.string().url().optional(),
    githubUrl: z.string().url().optional(),
    demoUrl: z.string().url().optional(),
    videoUrl: z.string().url().optional(),
    technologies: z.array(z.string()),
    category: z.enum(['web', 'mobile', 'desktop', 'ai', 'data', 'other']),
    featured: z.boolean().default(false),
    images: z.array(z.string()).optional(),
    codeSnippets: z.array(z.object({
      language: z.string(),
      title: z.string(),
      code: z.string(),
    })).optional(),
  }),
});

export const collections = {
  'authors': authorCollection,
  'blog': blogCollection,
  'portfolio': portfolioCollection,
};