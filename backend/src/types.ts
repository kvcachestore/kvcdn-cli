import { z } from "zod";

export const UploadMetaSchema = z.object({
  name: z.string().min(1),
  size_bytes: z.number().int().nonnegative(),
  sha256: z.string().regex(/^[0-9a-fA-F]{64}$/, "expected 64 hex chars"),
  dtype: z.string().min(1),
  storage_dtype: z.string().min(1).optional(),
  num_tokens: z.number().int().nonnegative(),
  num_layers: z.number().int().nonnegative(),
  quantized: z.boolean(),
  visibility: z.enum(["public", "private"]).default("private"),
});

export type UploadMeta = z.infer<typeof UploadMetaSchema>;