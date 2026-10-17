import React from 'react';
import { getAllPostSummaries } from '@/lib/posts';
import UnifiedHome from '@/components/UnifiedHome';

export default function HomePage() {
  const posts = getAllPostSummaries();
  return <UnifiedHome posts={posts} />;
}
