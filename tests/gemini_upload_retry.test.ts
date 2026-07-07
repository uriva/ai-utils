import { assertEquals } from "@std/assert";
import {
  injectRawUploadBlobToGemini,
  uploadBlobToGemini,
} from "../src/gemini.ts";
import { geminiUploadJsonParseErrorMessage } from "../src/utils.ts";

const uploadResult = {
  geminiUri: "https://generativelanguage.googleapis.com/v1beta/files/ok",
  mimeType: "application/pdf",
};

Deno.test(
  "uploadBlobToGemini recovers when the Gemini SDK throws a transient JSON-parse error",
  async () => {
    let attempts = 0;
    const flakyUpload = (_blob: Blob, _mimeType: string) => {
      attempts++;
      if (attempts === 1) {
        return Promise.reject(
          new SyntaxError(geminiUploadJsonParseErrorMessage),
        );
      }
      return Promise.resolve(uploadResult);
    };
    const result = await injectRawUploadBlobToGemini(() => flakyUpload)(() =>
      uploadBlobToGemini(new Blob(["x"]), "application/pdf")
    )();
    assertEquals(result, uploadResult);
    assertEquals(
      attempts,
      2,
      "the upload must be retried after the transient SyntaxError and then succeed",
    );
  },
);
