# FFmpeg runtime archive

The GitHub source repository does not commit the FFmpeg archive because it is a
large third-party binary. Before running `install_ffmpeg_native.ps1`, place this
file in this directory:

```text
ffmpeg-8.0.1-full_build-shared.zip
```

Verified local artifact:

- size: `94,371,734` bytes
- SHA-256: `e4a40b46e3a3f3e5f2ed28352bd1fd6c733ae3808a8743682ef7f0d4e20c7d51`
- build distributor: <https://www.gyan.dev/ffmpeg/builds/>

An offline delivery bundle may include this exact archive. Review the upstream
FFmpeg and build-distributor license terms before redistributing it.
