import { NavbarComp } from "@/components/Navbar";
import { BackgroundLines } from "@/components/ui/background-lines";


export default async function Index() {
  return (
    <>
   
    <div className="relative z-10">
      <BackgroundLines className="flex items-center justify-center w-full flex-col px-4 pt-32">
        <h2 className="bg-clip-text text-transparent text-center bg-gradient-to-b from-neutral-900 to-neutral-700 dark:from-neutral-600 dark:to-white text-2xl md:text-4xl lg:text-7xl font-sans py-2 md:py-10 relative z-20 font-bold tracking-tight">
          Prohibited Action Detection System<br />
        </h2>
        <p className="max-w-xl mx-auto text-sm md:text-lg text-neutral-700 dark:text-neutral-400 text-center">
          Preventing Problems Before They Start.
        </p>
      </BackgroundLines>
   
    {/* <div>
      <h1>Live Video Stream</h1>
      <img src="http://localhost:5000/video_feed" alt="Live Video" />
    </div> */}
    </div>
  </>
  );
}
