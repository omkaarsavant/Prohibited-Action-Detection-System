// import { NavbarComp } from "@/components/Navbar";
// import { BackgroundLines } from "@/components/ui/background-lines";


// export default async function Index() {
//   return (
//     <>
   
//     <div className="relative z-10">
//       <BackgroundLines className="flex items-center justify-center w-full flex-col px-4 pt-32">
//         <h2 className="bg-clip-text text-transparent text-center bg-gradient-to-b from-neutral-900 to-neutral-700 dark:from-neutral-600 dark:to-white text-2xl md:text-4xl lg:text-7xl font-sans py-2 md:py-10 relative z-20 font-bold tracking-tight">
//           Prohibited Action Detection System<br />
//         </h2>
//         <p className="max-w-xl mx-auto text-sm md:text-lg text-neutral-700 dark:text-neutral-400 text-center">
//           Preventing Problems Before They Start.
//         </p>
//       </BackgroundLines>
   
//     {/* <div>
//       <h1>Live Video Stream</h1>
//       <img src="http://localhost:5000/video_feed" alt="Live Video" />
//     </div> */}
//     </div>
//   </>
//   );
// }


'use client';

import { useSearchParams } from 'next/navigation';
import { useEffect, useState } from 'react';
import { NavbarComp } from "@/components/Navbar";
import { BackgroundLines } from "@/components/ui/background-lines";
import { Button } from '@/components/ui/button';
import Link from 'next/link';
import { supabase } from '@/lib/supabaseClient';

export default function Index() {
  const searchParams = useSearchParams();
  const [toastMsg, setToastMsg] = useState('');
  const [isLoggedIn, setIsLoggedIn] = useState(false);

  useEffect(() => {
    const checkLogin = async () => {
      const { data: { session } } = await supabase.auth.getSession();
      setIsLoggedIn(!!session);
    };
    checkLogin();
  }, []);

  useEffect(() => {
    const toast = searchParams.get('toast');
    if (toast === 'loggedin') {
      setToastMsg('Logged in successfully!');
      setTimeout(() => setToastMsg(''), 3000);
    }
  }, [searchParams]);

  return (
    <>
      {toastMsg && (
        <div className="fixed top-20 right-5 z-50 px-4 py-2 bg-green-600 text-white rounded shadow">
          {toastMsg}
        </div>
      )}

      <div className="relative z-10 flex flex-col items-center">
        <NavbarComp />
        <BackgroundLines className="flex items-center justify-center w-full flex-col px-4 pt-32">
          <h2 className="bg-clip-text text-transparent text-center bg-gradient-to-b from-neutral-900 to-neutral-700 dark:from-neutral-600 dark:to-white text-2xl md:text-4xl lg:text-7xl font-sans py-2 md:py-10 relative z-20 font-bold tracking-tight">
            Prohibited Action Detection System<br />
          </h2>
          <p className="max-w-xl mx-auto text-sm md:text-lg text-neutral-700 dark:text-neutral-400 text-center">
            Preventing Problems Before They Start.
          </p>
        </BackgroundLines>

        {!isLoggedIn && !toastMsg && (
          <Link href='/login' className='relative z-100 -top-16'>
            <Button>Login</Button>
          </Link>
        )}
      </div>
    </>
  );
}

