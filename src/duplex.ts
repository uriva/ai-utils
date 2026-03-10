export type DuplexMessage =
  | { type: "text"; text: string; from: string }
  | {
    type: "audio";
    chunks: Array<{ mimeType: string; dataBase64: string }>;
    from: string;
  }
  | { type: "close"; from: string }
  | { type: "flush"; from: string };

export type DuplexEndpoint = {
  sendData: (message: DuplexMessage) => Promise<void>;
  onData: (handler: (message: DuplexMessage) => void | Promise<void>) => void;
};

const endpoint = () => {
  let handler: ((message: DuplexMessage) => void | Promise<void>) | undefined;
  const queued: DuplexMessage[] = [];
  return {
    setHandler: (next: (message: DuplexMessage) => void | Promise<void>) => {
      handler = next;
      const pending = queued.splice(0);
      Promise.all(pending.map(handler));
    },
    deliver: async (message: DuplexMessage) => {
      if (!handler) {
        queued.push(message);
        return;
      }
      await handler(message);
    },
  };
};

export const createDuplexPair = (): {
  left: DuplexEndpoint;
  right: DuplexEndpoint;
} => {
  const left = endpoint();
  const right = endpoint();
  return {
    left: {
      sendData: (message: DuplexMessage) => right.deliver(message),
      onData: (handler: (message: DuplexMessage) => void | Promise<void>) => {
        left.setHandler(handler);
      },
    },
    right: {
      sendData: (message: DuplexMessage) => left.deliver(message),
      onData: (handler: (message: DuplexMessage) => void | Promise<void>) => {
        right.setHandler(handler);
      },
    },
  } satisfies { left: DuplexEndpoint; right: DuplexEndpoint };
};
