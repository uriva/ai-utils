import type { HistoryEvent } from "./agent.ts";
import type { DuplexMessage } from "./duplex.ts";

const wavHeader = (dataBytes: number, sampleRate: number, numChannels = 1) => {
  const bytesPerSample = 2;
  const byteRate = sampleRate * numChannels * bytesPerSample;
  const blockAlign = numChannels * bytesPerSample;
  const buffer = new ArrayBuffer(44);
  const view = new DataView(buffer);
  const writeString = (offset: number, value: string) => {
    for (let i = 0; i < value.length; i++) {
      view.setUint8(offset + i, value.charCodeAt(i));
    }
  };
  writeString(0, "RIFF");
  view.setUint32(4, 36 + dataBytes, true);
  writeString(8, "WAVE");
  writeString(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, numChannels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, byteRate, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, 16, true);
  writeString(36, "data");
  view.setUint32(40, dataBytes, true);
  return new Uint8Array(buffer);
};

const pcmToWav = (pcm: Uint8Array, sampleRate: number) => {
  const header = wavHeader(pcm.length, sampleRate);
  const wav = new Uint8Array(header.length + pcm.length);
  wav.set(header, 0);
  wav.set(pcm, header.length);
  return wav;
};

const base64ToBytes = (dataBase64: string) =>
  Uint8Array.from(atob(dataBase64), (char) => char.charCodeAt(0));

const concatBytes = (chunks: Uint8Array[]) => {
  const total = chunks.reduce((sum, chunk) => sum + chunk.length, 0);
  const merged = new Uint8Array(total);
  let offset = 0;
  for (const chunk of chunks) {
    merged.set(chunk, offset);
    offset += chunk.length;
  }
  return merged;
};

const silence24k = (ms: number) =>
  new Uint8Array(Math.floor((24000 * 2 * ms) / 1000));

export const createAudioArtifactsWriter = (dir: URL) => {
  const completedTurns: Uint8Array[] = [];
  const summaries: string[] = [];
  const summarize = (event: HistoryEvent) => {
    if (event.type === "tool_call") return `tool_call ${event.name}`;
    if (event.type === "tool_result") {
      return `tool_result ${event.result}`;
    }
    if (event.type === "participant_edit_message") {
      return `participant_edit_message ${event.name}: ${event.text}`;
    }
    if (event.type === "participant_utterance") {
      return `participant_utterance ${event.name}: ${event.text}`;
    }
    if (event.type === "own_utterance") return `own_utterance: ${event.text}`;
    return event.type;
  };
  return async (event: HistoryEvent) => {
    summaries.push(summarize(event));
    await Deno.writeTextFile(
      new URL("summary.txt", dir),
      summaries.join("\n\n"),
    );
    if (event.type !== "own_utterance" || !event.attachments?.length) return;
    const pcm = concatBytes(
      event.attachments.flatMap((attachment) =>
        attachment.kind === "inline"
          ? [base64ToBytes(attachment.dataBase64)]
          : []
      ),
    );
    completedTurns.push(pcm);
    const spacedTurns = completedTurns.flatMap((turn, index) =>
      index === completedTurns.length - 1 ? [turn] : [turn, silence24k(350)]
    );
    await Deno.writeFile(
      new URL("conversation.wav", dir),
      pcmToWav(concatBytes(spacedTurns), 24000),
    );
  };
};

export const createTransportAudioRecorder = (dir: URL) => {
  const startedAt = Date.now();
  const targetSampleRate = 24000;
  let finalDurationMs = 0;
  const clips: Array<{ startMs: number; pcm: Int16Array }> = [];
  const parseSampleRate = (mimeType: string) => {
    const match = mimeType.match(/rate=(\d+)/);
    return match ? Number(match[1]) : targetSampleRate;
  };
  const pcmDurationMs = (samples: number, sampleRate: number) =>
    Math.round((samples / sampleRate) * 1000);
  const bytesToInt16 = (pcm: Uint8Array) =>
    new Int16Array(
      pcm.buffer.slice(pcm.byteOffset, pcm.byteOffset + pcm.byteLength),
    );
  const resampleToTarget = (pcm: Int16Array, sampleRate: number) => {
    if (sampleRate === targetSampleRate) return pcm;
    const outLength = Math.floor(pcm.length * targetSampleRate / sampleRate);
    const output = new Int16Array(outLength);
    for (let i = 0; i < outLength; i++) {
      output[i] = pcm[Math.floor(i * sampleRate / targetSampleRate)] ?? 0;
    }
    return output;
  };
  const persist = async () => {
    const totalFrames = Math.ceil(finalDurationMs * targetSampleRate / 1000);
    const mixed = new Int32Array(totalFrames);
    for (const clip of clips) {
      const startFrame = Math.floor(clip.startMs * targetSampleRate / 1000);
      for (
        let i = 0;
        i < clip.pcm.length && startFrame + i < mixed.length;
        i++
      ) {
        mixed[startFrame + i] += clip.pcm[i];
      }
    }
    const clamped = new Int16Array(totalFrames);
    for (let i = 0; i < mixed.length; i++) {
      clamped[i] = Math.max(-32768, Math.min(32767, mixed[i]));
    }
    await Deno.writeFile(
      new URL("conversation.wav", dir),
      pcmToWav(new Uint8Array(clamped.buffer), targetSampleRate),
    );
  };
  return async (message: DuplexMessage) => {
    const now = Date.now();
    if (message.type === "close") {
      finalDurationMs = Math.max(finalDurationMs, now - startedAt);
      await persist();
      return;
    }
    if (message.type !== "audio") return;
    const pcm = concatBytes(
      message.chunks.map((chunk) => base64ToBytes(chunk.dataBase64)),
    );
    const sampleRate = parseSampleRate(
      message.chunks[0]?.mimeType ?? "audio/pcm;rate=24000",
    );
    const resampled = resampleToTarget(bytesToInt16(pcm), sampleRate);
    const startMs = now - startedAt;
    clips.push({ startMs, pcm: resampled });
    finalDurationMs = Math.max(
      finalDurationMs,
      startMs + pcmDurationMs(resampled.length, targetSampleRate),
    );
    await persist();
  };
};
