import React from 'react';
import { getAllPosts } from '@/lib/posts';
import UnifiedHome from '@/components/UnifiedHome';

export default function HomePage() {
  const posts = getAllPosts();
  return <UnifiedHome posts={posts} />;
}
