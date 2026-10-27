#!/bin/bash
# Install the 5 s pad link timeout: a udev rule that, on every xpadneo bind,
# starts crawler-pad-lst@<mac>.service, which runs deploy/pad-lst.sh as root.
# Independent of crawler.service; it works with the car's service stopped.
#
#   sudo ./deploy/install-pad-lst.sh
#   journalctl -u 'crawler-pad-lst@*'      # one line per pad connection
#
# Everything is copied out of the checkout, so moving or pulling the checkout
# never breaks it. It also applies the timeout to any pad already connected,
# because the rule only sees the next bind.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ $EUID -ne 0 ]]; then
  echo "Needs root: sudo $0" >&2
  exit 1
fi
command -v hcitool >/dev/null || echo "WARNING: no hcitool; the rule will install but cannot set anything" >&2

install -m 0755 "$HERE/pad-lst.sh"                /usr/local/sbin/crawler-pad-lst
install -m 0644 "$HERE/crawler-pad-lst@.service"  /etc/systemd/system/crawler-pad-lst@.service
install -m 0644 "$HERE/90-crawler-pad-lst.rules"  /etc/udev/rules.d/90-crawler-pad-lst.rules
systemctl daemon-reload
udevadm control --reload-rules

echo "Installed /usr/local/sbin/crawler-pad-lst, crawler-pad-lst@.service, 90-crawler-pad-lst.rules"

# Pads connected right now: every bluetooth HID device xpadneo has bound.
for dev in /sys/bus/hid/drivers/xpadneo/0005:*; do
  [[ -e "$dev" ]] || continue
  mac="$(sed -n 's/^HID_UNIQ=//p' "$dev/uevent")"
  [[ -n "$mac" ]] || continue
  systemctl start "crawler-pad-lst@$mac.service" || true
  journalctl -u "crawler-pad-lst@$mac.service" -n 5 -o cat --no-pager | grep "link supervision" | tail -n 1 || true
done
