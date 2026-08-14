import Link from "next/link";
import {
  deleteVerificationToken,
  getVerificationToken,
  updateUserEmailVerified,
} from "@/lib/db/queries";

export default async function VerifyEmailPage({
  searchParams,
}: {
  searchParams: Promise<{ token?: string }>;
}) {
  const resolvedParams = await searchParams;
  const { token } = resolvedParams;

  if (!token) {
    return (
      <div className="flex flex-col space-y-4 text-center">
        <h1 className="text-2xl font-semibold tracking-tight text-red-500">
          Invalid Verification Link
        </h1>
        <p className="text-sm text-muted-foreground">
          The link is missing the verification token.
        </p>
      </div>
    );
  }

  const verificationToken = await getVerificationToken(token);

  if (!verificationToken) {
    return (
      <div className="flex flex-col space-y-4 text-center">
        <h1 className="text-2xl font-semibold tracking-tight text-red-500">
          Invalid Token
        </h1>
        <p className="text-sm text-muted-foreground">
          The verification token is invalid or has already been used.
        </p>
      </div>
    );
  }

  if (new Date() > new Date(verificationToken.expires)) {
    await deleteVerificationToken(verificationToken.identifier, token);
    return (
      <div className="flex flex-col space-y-4 text-center">
        <h1 className="text-2xl font-semibold tracking-tight text-red-500">
          Token Expired
        </h1>
        <p className="text-sm text-muted-foreground">
          The verification link has expired. Please register again or request a
          new link.
        </p>
      </div>
    );
  }

  // Token is valid
  await updateUserEmailVerified(verificationToken.identifier);
  await deleteVerificationToken(verificationToken.identifier, token);

  return (
    <div className="flex flex-col space-y-4 text-center">
      <h1 className="text-2xl font-semibold tracking-tight text-green-500">
        Email Verified!
      </h1>
      <p className="text-sm text-muted-foreground">
        Your email address has been successfully verified. You can now sign in.
      </p>
      <Link
        className="mt-4 text-sm font-medium text-primary hover:underline"
        href="/login"
      >
        Continue to Sign In
      </Link>
    </div>
  );
}
