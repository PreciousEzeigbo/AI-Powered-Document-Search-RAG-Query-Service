import RAGChat from '@/components/RAGChat';

export const metadata = {
  title: 'RAG Chat',
  description: 'Chat with your documents using retrieval-augmented generation',
};

export default function Home() {
  return <RAGChat />;
}
