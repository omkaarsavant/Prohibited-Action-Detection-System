
"use client";

import Link from "next/link";
import { Button } from "./ui/button";

export function NavbarComp() {
  return (
    <div className="w-[100%] px-8 md:px-16 lg:px-32 py-4 shadow-md flex items-center justify-between z-10 fixed top-0 bg-white">
        <div className="text-2xl  font-bold ">
            Spot the Unseen
        </div>

        <ul className="flex items-center gap-8">
            <li className="cursor-pointer hover:text-gray-600 font-semibold hover:scale(125) transition-all">
                <Link href="/video-feed">Monitoring System</Link>
            </li>
            <li className="cursor-pointer hover:text-gray-600 hover:scale(125) font-semibold transition-all">
                <Link href="/registration">Register Prisoner</Link>
            </li>
            {/* <li className="cursor-pointer hover:text-gray-600 hover:scale(125) font-semibold transition-all">
                <Link href="/action-detection">Action Detection</Link>
            </li> */}
            {/* <li className="cursor-pointer hover:text-gray-600 hover:scale(125) font-semibold transition-all">
                <Link href="/video-feed">
                    <Button>Login/Signup</Button>
                </Link>
            </li> */}
        </ul>
    </div>
  );
}
