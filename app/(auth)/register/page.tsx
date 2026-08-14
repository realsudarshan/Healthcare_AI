"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useSession } from "next-auth/react";
import { useActionState, useEffect, useState } from "react";
import { AuthForm } from "@/components/chat/auth-form";
import { SubmitButton } from "@/components/chat/submit-button";
import { toast } from "@/components/chat/toast";
import { type RegisterActionState, register } from "../actions";

export default function Page() {
  const router = useRouter();
  const [email, setEmail] = useState("");
  const [isSuccessful, setIsSuccessful] = useState(false);

  const [state, formAction] = useActionState<RegisterActionState, FormData>(
    register,
    { status: "idle" }
  );

  const { update: updateSession } = useSession();

  // biome-ignore lint/correctness/useExhaustiveDependencies: router and updateSession are stable refs
  useEffect(() => {
    if (state.status === "user_exists") {
      toast({
        description: state.message || "Account already exists!",
        type: "error",
      });
    } else if (state.status === "failed") {
      toast({ description: "Failed to create account!", type: "error" });
    } else if (state.status === "invalid_data") {
      toast({
        description: state.message || "Failed validating your submission!",
        type: "error",
      });
    } else if (state.status === "success") {
      toast({ description: "Account created!", type: "success" });
      setIsSuccessful(true);
      updateSession();
      router.refresh();
    } else if (state.status === "requires_verification") {
      toast({
        description:
          state.message || "Please check your email to verify your account.",
        type: "success",
      });
      setIsSuccessful(true);
    }
  }, [state.status, state.message]);

  const handleSubmit = (formData: FormData) => {
    setEmail(formData.get("email") as string);
    formAction(formData);
  };

  if (state.status === "requires_verification") {
    return (
      <div className="flex flex-col space-y-4 text-center">
        <h1 className="text-2xl font-semibold tracking-tight">
          Check your email
        </h1>
        <p className="text-sm text-muted-foreground">
          We've sent a verification link to{" "}
          <span className="font-medium text-foreground">{email}</span>.
        </p>
        <p className="text-sm text-muted-foreground">
          Please click the link to verify your account before signing in.
        </p>
        <Link
          className="mt-4 text-sm font-medium text-primary hover:underline"
          href="/login"
        >
          Return to sign in
        </Link>
      </div>
    );
  }

  return (
    <>
      <h1 className="text-2xl font-semibold tracking-tight">Create account</h1>
      <p className="text-sm text-muted-foreground">
        Get started with Healthcare Support
      </p>
      <AuthForm action={handleSubmit} defaultEmail={email}>
        <SubmitButton isSuccessful={isSuccessful}>Sign up</SubmitButton>
        <p className="text-center text-[13px] text-muted-foreground">
          {"Have an account? "}
          <Link
            className="text-foreground underline-offset-4 hover:underline"
            href="/login"
          >
            Sign in
          </Link>
        </p>
      </AuthForm>
    </>
  );
}
