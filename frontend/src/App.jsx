import Layout from './components/Layout';
import { Toaster } from 'sonner';

export default function App() {
  return (
    <>
      <Toaster position="top-right" richColors />
      <Layout />
    </>
  );
}
