This is a [Next.js](https://nextjs.org) project bootstrapped with [`create-next-app`](https://github.com/vercel/next.js/tree/canary/packages/create-next-app).

## Getting Started

First, run the development server:

```bash
npm run dev
# or
yarn dev
# or
pnpm dev
# or
bun dev
```

Open [http://localhost:3000](http://localhost:3000) with your browser to see the result.

You can start editing the page by modifying `app/page.js`. The page auto-updates as you edit the file.

This project uses [`next/font`](https://nextjs.org/docs/app/building-your-application/optimizing/fonts) to automatically optimize and load [Geist](https://vercel.com/font), a new font family for Vercel.

## Learn More

To learn more about Next.js, take a look at the following resources:

- [Next.js Documentation](https://nextjs.org/docs) - learn about Next.js features and API.
- [Learn Next.js](https://nextjs.org/learn) - an interactive Next.js tutorial.

You can check out [the Next.js GitHub repository](https://github.com/vercel/next.js) - your feedback and contributions are welcome!

## Deploy on Vercel

The easiest way to deploy your Next.js app is to use the [Vercel Platform](https://vercel.com/new?utm_medium=default-template&filter=next.js&utm_source=create-next-app&utm_campaign=create-next-app-readme) from the creators of Next.js.

Check out our [Next.js deployment documentation](https://nextjs.org/docs/app/building-your-application/deploying) for more details.

{'start': np.float64(13.779999999999998), 'end': np.float64(17.22), 'sentence': "I'll miss you, Chihiro. Your best friend, Rumi.", 'translate': 'سأفتقدك يا (تشيهيرو) صديقك المفضل (رومي)', 'speaker': 'SPEAKER_00'}
{'start': np.float64(17.76), 'end': np.float64(20.72), 'sentence': "Chihiro? Chihiro, we're almost there.", 'translate': 'شيهيرو؟ شيهيرو، نحن تقريبا هناك.', 'speaker': 'SPEAKER_01'}
{'start': np.float64(22.86), 'end': np.float64(28.1), 'sentence': "This really is in the middle of nowhere. I'm gonna have to go to the next town to shop.", 'translate': 'هذا حقاً في وسط اللا مكان سأذهب إلى البلدة التالية لأتسوق', 'speaker': 'SPEAKER_05'}
{'start': np.float64(28.1), 'end': np.float64(30.12), 'sentence': "We'll just have to learn to like it.", 'translate': 'علينا فقط أن نتعلم أن نحبه', 'speaker': 'SPEAKER_02'}
{'start': np.float64(33.36), 'end': np.float64(36.52), 'sentence': "Look, Chihiro, there's your new school. Looks great, doesn't it?", 'translate': 'انظر يا (تشيهيرو)، هذه مدرستك الجديدة تبدو رائعة، أليس كذلك؟', 'speaker': 'SPEAKER_03'}
{'start': np.float64(37.86), 'end': np.float64(39.58), 'sentence': "It doesn't look so bad.", 'translate': 'لا يبدو الأمر سيئاً جداً', 'speaker': 'SPEAKER_04'}
{'start': np.float64(47.5), 'end': np.float64(49.98), 'sentence': "It's gonna stink. I like...", 'translate': 'إنها ستنتن، أحب...', 'speaker': 'SPEAKER_00'}
All resources cleaned up
[Transcriber] Sending result: This really is in the middle of nowhere. I'm gonna...
[Transcriber] Sending result: We'll just have to learn to like it....
[Transcriber] Sending result: Look, Chihiro, there's your new school. Looks grea...
[Transcriber] Sending result: It doesn't look so bad....
[Transcriber] Sending result: It's gonna stink. I like......
[Transcriber] Transcription process completed
[Transcriber] Processed 7 results from transcription
[Transcriber] Completed chunk 0
[Transcriber] Waiting for chunk 1...
[Transcriber] Received item: <class 'tuple'>
[Transcriber] Processing chunk 1 at 49.00s
[Transcriber] Started process for chunk 1
Sent chunk for time 343.00s
Chunk collection took 0.000189 seconds
C:\Users\molip\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\LocalCache\local-packages\Python311\site-packages\pyannote\audio\models\blocks\pooling.py:104: UserWarning: std(): degrees of freedom is <= 0. Correction should be strictly less than the reduction factor (input numel divided by output numel). (Triggered internally at C:\actions-runner\_work\pytorch\pytorch\pytorch\aten\src\ATen\native\ReduceOps.cpp:1839.)
  std = sequences.std(dim=-1, correction=1)
Starting transcription process for iteration 1
Silero model loaded in 0.156070 seconds
Speech timestamps computed in 0.946250 seconds
Found 6 speech segments
Speech probability analysis completed in 0.742293 seconds, highest prob: 0.9994528889656067
Silero model unloaded
Whisper model loaded in 4.314921 seconds
Transcription completed in 0.288335 seconds
Whisper model unloaded
Pyannote pipeline loaded in 2.646844 seconds
Diarization completed in 7.738952 seconds
Processing segment: ' I liked my old school.'
Speech exists: Y
Processing segment: ' Ah!'
Speech exists: Y
Processing segment: ' Ah!'
Speech exists: Y
Processing segment: ' Mom!'
Speech exists: Y
Processing segment: ' My flowers are dying!'
Speech exists: Y
Processing segment: ' I told you not to smother them like that.'
Speech exists: Y
Processing segment: ' We'll put them in water when we get to our new home.'
Speech exists: Y
Processing segment: ' I finally get a bouquet, and it's a goodbye present.'
Speech exists: Y
Processing segment: ' That's depressing.'
Speech exists: Y
Processing segment: ' Daddy bought you a rose for your birthday.'
Speech exists: Y
Processing segment: ' Don't you remember?'
Speech exists: Y
Processing segment: ' Yeah, one.'
Speech exists: Y
Processing segment: ' Just one rose isn't a bouquet.'
Speech exists: Y
Processing segment: ' Hold on to your card.'
Speech exists: Y
Processing segment: ' I'm opening the window.'
Speech exists: Y
Processing segment: ' And quit whining.'
Speech exists: Y
Processing segment: ' It's fun to move to a new place.'
Speech exists: Y
Processing segment: ' It's an adventure.'
Speech exists: Y
[DEBUG] Total words collected: 87
Speaker SPEAKER_03: 0.5s - 2.0s (chunk-relative)
Speaker SPEAKER_03: 5.2s - 7.9s (chunk-relative)
Speaker SPEAKER_02: 8.3s - 10.3s (chunk-relative)
Speaker SPEAKER_02: 11.0s - 13.1s (chunk-relative)
Speaker SPEAKER_03: 14.9s - 18.9s (chunk-relative)
Speaker SPEAKER_01: 18.5s - 21.9s (chunk-relative)
Speaker SPEAKER_03: 22.3s - 22.3s (chunk-relative)
Speaker SPEAKER_00: 22.3s - 27.9s (chunk-relative)
Speaker SPEAKER_00: 28.0s - 32.8s (chunk-relative)
Pyannote model unloaded
[DEBUG] Diarization segments (chunk-relative):
  0: SPEAKER_03 [0.50s-1.95s]
  1: SPEAKER_03 [5.25s-7.86s]
  2: SPEAKER_02 [8.28s-10.31s]
  3: SPEAKER_02 [10.98s-13.06s]
  4: SPEAKER_03 [14.88s-18.90s]
  5: SPEAKER_01 [18.51s-21.87s]
  6: SPEAKER_03 [22.26s-22.27s]
  7: SPEAKER_00 [22.27s-27.87s]
  8: SPEAKER_00 [28.01s-32.79s]
[DEBUG] Word: 'I' [0.00s-0.60s] (chunk-relative)
  vs Speaker SPEAKER_03 [0.50s-1.95s]: overlap=0.10s, distance=0.20s
  vs Speaker SPEAKER_03 [5.25s-7.86s]: overlap=0.00s, distance=4.95s
  vs Speaker SPEAKER_02 [8.28s-10.31s]: overlap=0.00s, distance=7.98s
  vs Speaker SPEAKER_02 [10.98s-13.06s]: overlap=0.00s, distance=10.68s
  vs Speaker SPEAKER_03 [14.88s-18.90s]: overlap=0.00s, distance=14.58s
  vs Speaker SPEAKER_01 [18.51s-21.87s]: overlap=0.00s, distance=18.21s
  vs Speaker SPEAKER_03 [22.26s-22.27s]: overlap=0.00s, distance=21.96s
  vs Speaker SPEAKER_00 [22.27s-27.87s]: overlap=0.00s, distance=21.97s
  vs Speaker SPEAKER_00 [28.01s-32.79s]: overlap=0.00s, distance=27.71s
  -> Assigned: SPEAKER_03 (overlap=0.10s)
[DEBUG] Word: 'liked' [0.60s-0.86s] (chunk-relative)
  vs Speaker SPEAKER_03 [0.50s-1.95s]: overlap=0.26s, distance=0.00s
  vs Speaker SPEAKER_03 [5.25s-7.86s]: overlap=0.00s, distance=4.52s
  vs Speaker SPEAKER_02 [8.28s-10.31s]: overlap=0.00s, distance=7.55s
  vs Speaker SPEAKER_02 [10.98s-13.06s]: overlap=0.00s, distance=10.25s
  vs Speaker SPEAKER_03 [14.88s-18.90s]: overlap=0.00s, distance=14.15s
  vs Speaker SPEAKER_01 [18.51s-21.87s]: overlap=0.00s, distance=17.78s
  vs Speaker SPEAKER_03 [22.26s-22.27s]: overlap=0.00s, distance=21.53s
  vs Speaker SPEAKER_00 [22.27s-27.87s]: overlap=0.00s, distance=21.54s
  vs Speaker SPEAKER_00 [28.01s-32.79s]: overlap=0.00s, distance=27.28s
  -> Assigned: SPEAKER_03 (overlap=0.26s)
[DEBUG] Word: 'my' [0.86s-1.06s] (chunk-relative)
  vs Speaker SPEAKER_03 [0.50s-1.95s]: overlap=0.20s, distance=0.00s
  vs Speaker SPEAKER_03 [5.25s-7.86s]: overlap=0.00s, distance=4.29s
  vs Speaker SPEAKER_02 [8.28s-10.31s]: overlap=0.00s, distance=7.32s
  vs Speaker SPEAKER_02 [10.98s-13.06s]: overlap=0.00s, distance=10.02s
  vs Speaker SPEAKER_03 [14.88s-18.90s]: overlap=0.00s, distance=13.92s
  vs Speaker SPEAKER_01 [18.51s-21.87s]: overlap=0.00s, distance=17.55s
  vs Speaker SPEAKER_03 [22.26s-22.27s]: overlap=0.00s, distance=21.30s
  vs Speaker SPEAKER_00 [22.27s-27.87s]: overlap=0.00s, distance=21.31s
  vs Speaker SPEAKER_00 [28.01s-32.79s]: overlap=0.00s, distance=27.05s
  -> Assigned: SPEAKER_03 (overlap=0.20s)
[DEBUG] Word: 'old' [1.06s-1.36s] (chunk-relative)
  vs Speaker SPEAKER_03 [0.50s-1.95s]: overlap=0.30s, distance=0.00s
  vs Speaker SPEAKER_03 [5.25s-7.86s]: overlap=0.00s, distance=4.04s
  vs Speaker SPEAKER_02 [8.28s-10.31s]: overlap=0.00s, distance=7.07s
  vs Speaker SPEAKER_02 [10.98s-13.06s]: overlap=0.00s, distance=9.77s
  vs Speaker SPEAKER_03 [14.88s-18.90s]: overlap=0.00s, distance=13.67s
  vs Speaker SPEAKER_01 [18.51s-21.87s]: overlap=0.00s, distance=17.30s
  vs Speaker SPEAKER_03 [22.26s-22.27s]: overlap=0.00s, distance=21.05s
  vs Speaker SPEAKER_00 [22.27s-27.87s]: overlap=0.00s, distance=21.06s
  vs Speaker SPEAKER_00 [28.01s-32.79s]: overlap=0.00s, distance=26.80s
  -> Assigned: SPEAKER_03 (overlap=0.30s)
[DEBUG] Word: 'school.' [1.36s-1.78s] (chunk-relative)
  vs Speaker SPEAKER_03 [0.50s-1.95s]: overlap=0.42s, distance=0.00s
  vs Speaker SPEAKER_03 [5.25s-7.86s]: overlap=0.00s, distance=3.68s
  vs Speaker SPEAKER_02 [8.28s-10.31s]: overlap=0.00s, distance=6.71s
  vs Speaker SPEAKER_02 [10.98s-13.06s]: overlap=0.00s, distance=9.41s
  vs Speaker SPEAKER_03 [14.88s-18.90s]: overlap=0.00s, distance=13.31s
  vs Speaker SPEAKER_01 [18.51s-21.87s]: overlap=0.00s, distance=16.94s
  vs Speaker SPEAKER_03 [22.26s-22.27s]: overlap=0.00s, distance=20.69s
  vs Speaker SPEAKER_00 [22.27s-27.87s]: overlap=0.00s, distance=20.70s
  vs Speaker SPEAKER_00 [28.01s-32.79s]: overlap=0.00s, distance=26.44s
  -> Assigned: SPEAKER_03 (overlap=0.42s)
[DEBUG] Word: 'Ah!' [2.90s-3.08s] (chunk-relative)
  vs Speaker SPEAKER_03 [0.50s-1.95s]: overlap=0.00s, distance=1.04s
  vs Speaker SPEAKER_03 [5.25s-7.86s]: overlap=0.00s, distance=2.26s
  vs Speaker SPEAKER_02 [8.28s-10.31s]: overlap=0.00s, distance=5.29s
  vs Speaker SPEAKER_02 [10.98s-13.06s]: overlap=0.00s, distance=7.99s
  vs Speaker SPEAKER_03 [14.88s-18.90s]: overlap=0.00s, distance=11.89s
  vs Speaker SPEAKER_01 [18.51s-21.87s]: overlap=0.00s, distance=15.52s
  vs Speaker SPEAKER_03 [22.26s-22.27s]: overlap=0.00s, distance=19.27s
  vs Speaker SPEAKER_00 [22.27s-27.87s]: overlap=0.00s, distance=19.28s
  vs Speaker SPEAKER_00 [28.01s-32.79s]: overlap=0.00s, distance=25.02s
  -> Assigned: SPEAKER_03 (overlap=0.00s)
[DEBUG] Word: 'Ah!' [3.72s-4.08s] (chunk-relative)
  vs Speaker SPEAKER_03 [0.50s-1.95s]: overlap=0.00s, distance=1.95s
  vs Speaker SPEAKER_03 [5.25s-7.86s]: overlap=0.00s, distance=1.35s
  vs Speaker SPEAKER_02 [8.28s-10.31s]: overlap=0.00s, distance=4.38s
  vs Speaker SPEAKER_02 [10.98s-13.06s]: overlap=0.00s, distance=7.08s
  vs Speaker SPEAKER_03 [14.88s-18.90s]: overlap=0.00s, distance=10.98s
  vs Speaker SPEAKER_01 [18.51s-21.87s]: overlap=0.00s, distance=14.61s
  vs Speaker SPEAKER_03 [22.26s-22.27s]: overlap=0.00s, distance=18.36s
  vs Speaker SPEAKER_00 [22.27s-27.87s]: overlap=0.00s, distance=18.37s
  vs Speaker SPEAKER_00 [28.01s-32.79s]: overlap=0.00s, distance=24.11s