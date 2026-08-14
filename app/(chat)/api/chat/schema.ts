import { z } from "zod";

const textPartSchema = z.object({
  text: z.string().max(2000),
  type: z.enum(["text"]),
});

const filePartSchema = z.object({
  mediaType: z.enum(["image/jpeg", "image/png"]),
  name: z.string().min(1).max(500),
  type: z.enum(["file"]),
  url: z.url(),
});

const partSchema = z.union([textPartSchema, filePartSchema]);

const userMessageSchema = z
  .object({
    id: z.uuid(),
    parts: z.array(partSchema).min(1),
    role: z.enum(["user"]),
  })
  .refine(
    (message) =>
      message.parts.some(
        (part) =>
          part.type === "file" ||
          (part.type === "text" && part.text.trim().length > 0)
      ),
    { message: "Message must include text or an image attachment" }
  );

const toolApprovalMessageSchema = z.object({
  id: z.string(),
  parts: z.array(z.record(z.string(), z.unknown())),
  role: z.enum(["user", "assistant"]),
});

export const postRequestBodySchema = z.object({
  id: z.uuid(),
  message: userMessageSchema.optional(),
  messages: z.array(toolApprovalMessageSchema).optional(),
  selectedChatModel: z.string(),
  selectedVisibilityType: z.enum(["public", "private"]),
});

export type PostRequestBody = z.infer<typeof postRequestBodySchema>;
