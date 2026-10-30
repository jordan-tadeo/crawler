#!/bin/bash
# Install the WiFi fallback on the Pi: sudo ./install.sh
#
# Creates the crawler-hotspot AP profile if it is missing (SSID jt-rc-1, WPA2,
# 10.42.0.1, autoconnect off) with a fresh random password, printed once.
# Client networks are plain NetworkManager profiles with autoconnect on; add
# them with the password on stdin so it never lands in a file, e.g.
#   nmcli connection add type wifi con-name office-wifi ifname wlan0 \
#     ssid "Office WiFi" wifi-sec.key-mgmt wpa-psk wifi-sec.psk "$(cat)" \
#     connection.autoconnect-priority 50
# Priorities in use: home 100, office-wifi 50, phone-hotspot 40.
set -eu
cd "$(dirname "$0")"

if ! nmcli -g connection.id connection show crawler-hotspot >/dev/null 2>&1; then
    p=$(tr -dc 'abcdefghjkmnpqrstuvwxyz23456789' </dev/urandom | head -c 16)
    nmcli connection add type wifi con-name crawler-hotspot ifname wlan0 ssid jt-rc-1 \
        802-11-wireless.mode ap 802-11-wireless.band bg 802-11-wireless.channel 6 \
        ipv4.method shared ipv4.addresses 10.42.0.1/24 ipv6.method disabled \
        wifi-sec.key-mgmt wpa-psk wifi-sec.proto rsn wifi-sec.pairwise ccmp \
        wifi-sec.group ccmp wifi-sec.pmf disable wifi-sec.psk "$p" \
        connection.autoconnect no >/dev/null
    echo "created crawler-hotspot, SSID jt-rc-1, password: $p"
    unset p
fi

install -m 755 crawler-wifi-fallback /usr/local/sbin/crawler-wifi-fallback
install -m 644 crawler-wifi-fallback.service /etc/systemd/system/crawler-wifi-fallback.service
systemctl daemon-reload
systemctl enable --now crawler-wifi-fallback.service
systemctl restart crawler-wifi-fallback.service
echo "crawler-wifi-fallback: $(systemctl is-active crawler-wifi-fallback.service)"
