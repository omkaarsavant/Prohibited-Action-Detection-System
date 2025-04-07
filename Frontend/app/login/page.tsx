// 'use client';

// import { useEffect, useState } from 'react';
// import { useRouter } from 'next/navigation';
// import { supabase } from '@/lib/supabaseClient';

// export default function AuthPage() {
//   const [email, setEmail] = useState('');
//   const [password, setPassword] = useState('');
//   const [isLogin, setIsLogin] = useState(true);
//   const router = useRouter();

//   useEffect(() => {
//     const checkSession = async () => {
//       const { data: { session } } = await supabase.auth.getSession();
//       if (session) router.push('/');
//     };
//     checkSession();
//   }, []);

//   const handleAuth = async () => {
//     let result;
  
//     if (isLogin) {
//       result = await supabase.auth.signInWithPassword({ email, password });
//     } else {
//       result = await supabase.auth.signUp({ email, password });
//     }
  
//     const { error } = result;
  
//     if (error) {
//       alert(error.message);
//     } else {
//       router.push('/');
//     }
//   };
  

//   return (
//     <div className="flex flex-col items-center justify-center min-h-screen px-4">
//       <h2 className="text-3xl font-bold mb-4">{isLogin ? 'Login' : 'Sign Up'}</h2>
//       <input
//         type="email"
//         placeholder="Email"
//         className="border p-2 mb-2 w-full max-w-sm"
//         value={email}
//         onChange={(e) => setEmail(e.target.value)}
//       />
//       <input
//         type="password"
//         placeholder="Password"
//         className="border p-2 mb-4 w-full max-w-sm"
//         value={password}
//         onChange={(e) => setPassword(e.target.value)}
//       />
//       <button
//         onClick={handleAuth}
//         className="bg-black text-white px-6 py-2 rounded-md"
//       >
//         {isLogin ? 'Login' : 'Sign Up'}
//       </button>
//       <p
//         className="mt-4 text-blue-600 cursor-pointer"
//         onClick={() => setIsLogin(!isLogin)}
//       >
//         {isLogin ? "Don't have an account? Sign Up" : "Already have an account? Login"}
//       </p>
//     </div>
//   );
// }

'use client';

import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import { supabase } from '@/lib/supabaseClient';

export default function AuthPage() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [isLogin, setIsLogin] = useState(true);
  const [loading, setLoading] = useState(false);
  const [toastMsg, setToastMsg] = useState('');
  const [toastType, setToastType] = useState<'success' | 'error'>('success');

  const router = useRouter();

  useEffect(() => {
    const checkSession = async () => {
      const { data: { session } } = await supabase.auth.getSession();
      if (session) router.push('/');
    };
    checkSession();
  }, []);

  const showToast = (message: string, type: 'success' | 'error') => {
    setToastMsg(message);
    setToastType(type);
    setTimeout(() => setToastMsg(''), 3000);
  };

  const handleAuth = async () => {
    setLoading(true);
    let result;

    if (isLogin) {
      result = await supabase.auth.signInWithPassword({ email, password });
    } else {
      result = await supabase.auth.signUp({ email, password });
    }

    const { error } = result;
    setLoading(false);

    if (error) {
      showToast(error.message, 'error');
    } else {
      if (!isLogin) {
        // showToast('Signup successful! Please confirm your email.', 'success');
        showToast('Signup successful!', 'success');
      } else {
        showToast('Login successful!', 'success');
        router.push('/?toast=loggedin');
      }
    }
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen px-4 relative">
      <h2 className="text-3xl font-bold mb-4">{isLogin ? 'Login' : 'Sign Up'}</h2>

      <input
        type="email"
        placeholder="Email"
        className="border p-2 mb-2 w-full max-w-sm rounded"
        value={email}
        onChange={(e) => setEmail(e.target.value)}
      />
      <input
        type="password"
        placeholder="Password"
        className="border p-2 mb-4 w-full max-w-sm rounded"
        value={password}
        onChange={(e) => setPassword(e.target.value)}
      />

      <button
        onClick={handleAuth}
        className="bg-black text-white px-6 py-2 rounded-md w-full max-w-sm flex items-center justify-center disabled:opacity-50"
        disabled={loading}
      >
        {loading ? (
          <span className="border-2 border-white border-t-transparent w-5 h-5 rounded-full animate-spin" />
        ) : (
          isLogin ? 'Login' : 'Sign Up'
        )}
      </button>

      <p
        className="mt-4 text-blue-600 cursor-pointer"
        onClick={() => setIsLogin(!isLogin)}
      >
        {isLogin ? "Don't have an account? Sign Up" : "Already have an account? Login"}
      </p>

      {/* Toast */}
      {toastMsg && (
        <div
          className={`fixed top-5 right-5 z-50 px-4 py-2 rounded-md text-white shadow-md transition-all duration-300
            ${toastType === 'success' ? 'bg-green-600' : 'bg-red-600'}`}
        >
          {toastMsg}
        </div>
      )}
    </div>
  );
}


