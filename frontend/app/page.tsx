import RAGChatClientShell from '@/components/RAGChatClientShell';

export const metadata = {
  title: 'RAG Chat',
  description: 'Chat with your documents using retrieval-augmented generation',
};

export const viewport = {
  width: 'device-width',
  initialScale: 1,
  maximumScale: 1,
};

export default function Home() {
  return <RAGChatClientShell />;
}
