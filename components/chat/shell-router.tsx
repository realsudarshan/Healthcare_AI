"use client";

import { usePathname } from "next/navigation";
import { ActiveChatProvider } from "@/hooks/use-active-chat";
import { ChatShell } from "./shell";

export function ChatShellRouter() {
  const pathname = usePathname();

  if (pathname === "/patients" || pathname.startsWith("/patients/")) {
    return null;
  }

  return (
    <ActiveChatProvider>
      <ChatShell />
    </ActiveChatProvider>
  );
}
