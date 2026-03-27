import re

with open('src/geminiLiveSession.ts', 'r') as f:
    content = f.read()

# I want to log the payload before sending it
content = content.replace(
    'ws.send(JSON.stringify({',
    'const __payload = {'
).replace(
    '      }));',
    '      };\n      console.log("SENDING:", JSON.stringify(__payload));\n      ws.send(JSON.stringify(__payload));'
)

with open('src/geminiLiveSession.ts', 'w') as f:
    f.write(content)
