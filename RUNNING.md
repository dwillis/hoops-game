# Running Hoops

Hoops is a **terminal program**. You don't double-click it like a normal app —
you open a terminal window and type a command to start it. Don't worry, it's
only a couple of steps. Pick your operating system below and follow along.

> **The program is named differently per platform.** When you download it from
> the [Releases page](https://github.com/dwillis/hoops-game/releases) you'll get
> one of these:
>
> | Your computer | File you download |
> |---------------|-------------------|
> | Mac           | `hoops-macos`     |
> | Windows       | `hoops-windows.exe` |
> | Linux         | `hoops-linux`     |
>
> The instructions below assume you've put the downloaded file in your
> **Downloads** folder, which is where browsers normally save things.

---

## 🍎 macOS

macOS protects you from running programs it hasn't seen before. Because Hoops
is a small free project and isn't signed by Apple, your Mac will refuse to open
it the first time and show a scary-looking warning. This is expected — here's
how to get past it safely.

### Step 1 — Open Terminal

Press `Cmd` (⌘) + `Space`, type **Terminal**, and press `Return`. A window with
a text prompt opens. You'll type commands here.

### Step 2 — Go to your Downloads folder

Type this and press `Return`:

```bash
cd ~/Downloads
```

### Step 3 — Allow the program to run

Copy and paste these two lines, pressing `Return` after each one:

```bash
chmod +x hoops-macos
xattr -d com.apple.quarantine hoops-macos
```

- The first line makes the file runnable.
- The second line removes the "downloaded from the internet" flag that macOS
  uses to block it (this is the **quarantine** flag). Running this is what stops
  the *"hoops-macos cannot be opened because the developer cannot be verified"*
  and *"hoops-macos is damaged and can't be opened"* warnings.

If you see `No such file or attribute`, that's fine — it just means the flag
wasn't there.

### Step 4 — Start the game

```bash
./hoops-macos play
```

The team picker appears and you're coaching.

### If you skipped Step 3 and got blocked

If you double-clicked the file or ran it and macOS blocked it, you can also
approve it through System Settings:

1. Try to open the program once (so macOS records the block).
2. Open **System Settings → Privacy & Security**.
3. Scroll down to the **Security** section. You'll see a message like
   *"hoops-macos was blocked to protect your Mac."*
4. Click **Open Anyway**, then confirm.

After that, the `./hoops-macos play` command in Step 4 will work.

---

## 🪟 Windows

### Step 1 — Open PowerShell

Click the **Start** menu, type **PowerShell**, and press `Enter`.

### Step 2 — Go to your Downloads folder

```powershell
cd ~\Downloads
```

### Step 3 — Start the game

```powershell
.\hoops-windows.exe play
```

### If Windows shows a blue "Windows protected your PC" box

Because Hoops isn't a widely-downloaded, signed app, Windows SmartScreen may
warn you. To run it anyway:

1. Click **More info** on the blue dialog.
2. Click the **Run anyway** button that appears.

This only happens the first time. Your antivirus may also quarantine the file —
if the program "disappears" after download, check your antivirus quarantine and
allow it.

---

## 🐧 Linux

### Step 1 — Open a terminal

Use your desktop's terminal app (often `Ctrl` + `Alt` + `T`).

### Step 2 — Go to your Downloads folder and make it runnable

```bash
cd ~/Downloads
chmod +x hoops-linux
```

### Step 3 — Start the game

```bash
./hoops-linux play
```

---

## Playing the game

Once it's running, here are the keys you'll use most:

| Key     | Action |
|---------|--------|
| `Space` | Advance one possession |
| `F`     | Fast-forward (hold) |
| `D`     | Cycle defensive scheme (man / zone / press) |
| `O`     | Cycle offensive scheme (balanced / hurry / three-point) |
| `U`     | Toggle substitution screen |
| `X`     | Toggle box score detail |
| `S`     | Save game |
| `L`     | Load game |

To jump straight into a specific matchup instead of using the picker, add team
names after `play`. For example, on a Mac:

```bash
./hoops-macos play --home "South Carolina" --away "Iowa" --season 2023-24
```

For the full list of game modes and options, see the
[README](README.md).

---

## Common questions

**Do I need to install Python?**
No. The downloaded file already contains everything it needs.

**Why does it look like a "virus" warning?**
It isn't one. macOS and Windows show these warnings for *any* program that
isn't signed by a paid developer certificate, even harmless hobby projects like
this one. The source code is fully public in this repository if you'd like to
inspect or build it yourself.

**Can I just double-click it?**
Not reliably. Hoops needs the `play` command (and sometimes other options) to
know what to do, so it's best to start it from a terminal as shown above.

**It says "permission denied."**
You skipped the `chmod +x` step. Run the `chmod` command for your platform
above, then try again.

**I'd rather build it myself.**
See the [Install from Source](README.md#install-from-source) section of the
README.
