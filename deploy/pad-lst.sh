#!/bin/bash
# Shorten the Bluetooth link supervision timeout for one game pad, and say
# what it reads back as. Installed as /usr/local/sbin/crawler-pad-lst and run
# as root by crawler-pad-lst@<mac>.service, which the udev rule starts every
# time xpadneo binds a pad. Nothing here touches the crawler service.
#
#   crawler-pad-lst 9c:aa:1b:b6:6b:34          # 8000 slots, 5 s
#   crawler-pad-lst 9c:aa:1b:b6:6b:34 16000    # any other value, in slots
#
# Why this exists: a pad sitting still sends no HID reports at all, so the only
# thing that notices a pad that died quietly is the link supervision timeout.
# Its default is 32000 slots, 20 s, and only when it runs out does the kernel
# drop the link and the input device, which is what the car's lost-pad path
# waits for. At 8000 slots that is 5 s. A slot is 0.625 ms.
#
# Why hcitool, deprecated as it is: it is the only tool on the car that sends
# HCI Write_Link_Supervision_Timeout. btmgmt has no command for it, and the
# LinkSupervisionTimeout key in /etc/bluetooth/main.conf only sets the
# kernel's def_br_lsto, which the kernel (6.6) stores and never sends to the
# controller. The value is per connection and gone on the next one, which is
# why this runs on every bind and not once at boot.
#
# The write needs the ACL handle, and the HID device can bind a moment before
# the handle answers, so it retries for a few seconds before giving up.

set -u

MAC="${1:?usage: crawler-pad-lst <mac> [slots]}"
SLOTS="${2:-8000}"
TRIES=20
DELAY=0.25

if ! [[ "$MAC" =~ ^([0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}$ ]]; then
  echo "not a Bluetooth address: $MAC" >&2
  exit 2
fi
if ! [[ "$SLOTS" =~ ^[0-9]+$ ]] || (( SLOTS < 1 || SLOTS > 65535 )); then
  echo "slots must be 1..65535, got $SLOTS" >&2
  exit 2
fi

HCITOOL="$(command -v hcitool || true)"
if [[ -z "$HCITOOL" ]]; then
  echo "no hcitool on this machine; $MAC keeps its default timeout" >&2
  exit 1
fi

# hcitool lst <mac> prints "Link supervision timeout: 8000 slots (5000.00 msec)".
read_slots() {
  "$HCITOOL" lst "$MAC" 2>/dev/null | sed -n 's/.*timeout: \([0-9][0-9]*\) slots.*/\1/p'
}

before=""
for ((i = 1; i <= TRIES; i++)); do
  [[ -n "$before" ]] || before="$(read_slots)"
  if "$HCITOOL" lst "$MAC" "$SLOTS" >/dev/null 2>&1; then
    after="$(read_slots)"
    if [[ "$after" == "$SLOTS" ]]; then
      ms=$(( SLOTS * 625 / 1000 ))
      echo "pad $MAC link supervision timeout ${after} slots (${ms} ms), was ${before:-unread}, try $i"
      exit 0
    fi
  fi
  sleep "$DELAY"
done

echo "pad $MAC link supervision timeout NOT set: wanted $SLOTS slots, reads ${after:-${before:-nothing}} after $TRIES tries" >&2
exit 1
