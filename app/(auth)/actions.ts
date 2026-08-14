"use server";

import { z } from "zod";

import { createUser, createVerificationToken, getUser } from "@/lib/db/queries";

import { signIn } from "./auth";

const passwordValidationSchema = z
  .string()
  .min(8, "Password must be at least 8 characters long.")
  .max(100, "Password cannot exceed 100 characters.")
  .regex(/[A-Z]/, "Password must contain at least one uppercase letter.")
  .regex(/[a-z]/, "Password must contain at least one lowercase letter.")
  .regex(/[0-9]/, "Password must contain at least one number.")
  .regex(
    /[^A-Za-z0-9]/,
    "Password must contain at least one special character (e.g. !@#$%^&*)."
  );

const loginFormSchema = z.object({
  email: z.email("Please enter a valid email address."),
  password: z.string().min(1, "Password is required."),
});

const registerFormSchema = z.object({
  email: z.email("Please enter a valid email address."),
  password: passwordValidationSchema,
});

export type LoginActionState = {
  status:
    | "idle"
    | "in_progress"
    | "success"
    | "failed"
    | "invalid_data"
    | "unverified_email";
  message?: string;
};

export const login = async (
  _: LoginActionState,
  formData: FormData
): Promise<LoginActionState> => {
  try {
    const validatedData = loginFormSchema.parse({
      email: formData.get("email"),
      password: formData.get("password"),
    });

    await signIn("credentials", {
      email: validatedData.email,
      password: validatedData.password,
      redirect: false,
    });

    return { status: "success" };
  } catch (error) {
    if (error instanceof z.ZodError) {
      const firstError =
        error.issues[0]?.message || "Invalid email or password format.";
      return { message: firstError, status: "invalid_data" };
    }

    if (
      error instanceof Error &&
      error.message.includes("Email not verified")
    ) {
      return {
        message: "Please verify your email before signing in.",
        status: "unverified_email",
      };
    }
    if (error && typeof error === "object" && "cause" in error) {
      const { cause } = error as any;
      const causeMessage =
        typeof cause === "string"
          ? cause
          : cause?.message || cause?.err?.message;
      if (causeMessage?.includes("Email not verified")) {
        return {
          message: "Please verify your email before signing in.",
          status: "unverified_email",
        };
      }
    }

    return { status: "failed" };
  }
};

export type RegisterActionState = {
  status:
    | "idle"
    | "in_progress"
    | "success"
    | "failed"
    | "user_exists"
    | "invalid_data"
    | "requires_verification";
  message?: string;
};

export const register = async (
  _: RegisterActionState,
  formData: FormData
): Promise<RegisterActionState> => {
  try {
    const validatedData = registerFormSchema.parse({
      email: formData.get("email"),
      password: formData.get("password"),
    });

    const [user] = await getUser(validatedData.email);

    if (user) {
      return {
        message: "An account with this email already exists.",
        status: "user_exists",
      } as RegisterActionState;
    }

    await createUser(validatedData.email, validatedData.password);

    const [verificationToken] = await createVerificationToken(
      validatedData.email
    );
    const verificationUrl = `${process.env.NEXT_PUBLIC_APP_URL || "http://localhost:3000"}/verify-email?token=${verificationToken.token}`;

    // Mocking email sending for now
    console.log(
      `[EMAIL MOCK] Please verify your email for ${validatedData.email}. URL: ${verificationUrl}`
    );

    return {
      message: "Please check your email to verify your account.",
      status: "requires_verification",
    };
  } catch (error) {
    if (error instanceof z.ZodError) {
      const firstError =
        error.issues[0]?.message ||
        "Password does not meet validation criteria.";
      return { message: firstError, status: "invalid_data" };
    }

    return { status: "failed" };
  }
};
