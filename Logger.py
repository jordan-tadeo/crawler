import logging
import os
from logging.handlers import RotatingFileHandler
from datetime import datetime

# This checkout, worked out from where this file sits rather than from where
# the process happens to be standing.
REPO_DIR = os.path.dirname(os.path.abspath(__file__))


def _prepare(path):
    '''Resolve a log path against the checkout and make sure its folder exists.

    Two separate things, both learned the hard way.

    The folder: `log/` holds nothing but generated CSVs, so it is gitignored,
    so it is not in a fresh clone -- and open(..., 'a') does not create a
    missing parent. The csv() call below sits in the 100 Hz control loop but
    only runs once a controller is connected and the state has changed, which
    is why a clone that cannot write it does not fall over at startup. It
    falls over a few seconds in, on the first state change, and with
    Restart=on-failure that is a restart loop. I patched it on the car with
    `mkdir -p log` once; this is so I never have to again.

    The checkout: the unit does set WorkingDirectory to this directory, so
    relative paths resolve correctly there today. But that is the unit's
    promise, not a property of the code -- run main.py from anywhere else, or
    change that line, and the logs quietly land somewhere else. Anchoring here
    means `log/vehicle_state_log.csv` is the same file whoever starts the car.
    An absolute path handed in is left exactly as it was given.
    '''

    path = os.path.join(REPO_DIR, path)
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    return path


class Logger:
    def __init__(self, log_file="app.log", max_size=10 * 1024 * 1024, backup_count=1):
        log_file = _prepare(log_file)

        self.logger = logging.getLogger("VehicleLogger")
        self.logger.setLevel(logging.INFO)

        # File handler with rotation
        file_handler = RotatingFileHandler(log_file, maxBytes=max_size, backupCount=backup_count)
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(subject)s - %(message)s')
        file_handler.setFormatter(file_formatter)

        # Console handler
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(subject)s - %(message)s')
        console_handler.setFormatter(console_formatter)

        # getLogger returns the same process-global logger every time, and
        # addHandler just appends -- so a second Logger() anywhere in the
        # process would give it a second file handler and a second console
        # handler, and every line after that would be written twice. Only one
        # of these is built today. This is so that stays true by accident.
        if not self.logger.handlers:
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)

    def log(self, level, subject, message):
        extra = {"subject": subject}
        if level == "info":
            self.logger.info(message, extra=extra)
        elif level == "warning":
            self.logger.warning(message, extra=extra)
        elif level == "error":
            self.logger.error(message, extra=extra)
        elif level == "debug":
            self.logger.debug(message, extra=extra)
        else:
            raise ValueError("Invalid log level. Use 'info', 'warning', 'error', or 'debug'.")

    def csv(self, csv_file, subject, message, max_size=10 * 1024 * 1024):
        import csv
        from datetime import datetime

        # Before anything looks at the path: anchor it to the checkout and
        # make its folder. Everything below -- the size check, both rotations,
        # the header read and the append -- works off this one resolved path.
        csv_file = _prepare(csv_file)

        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Check if the file exceeds the max size
        if os.path.exists(csv_file) and os.path.getsize(csv_file) > max_size:
            # Rotate the file by renaming it
            base, ext = os.path.splitext(csv_file)
            os.rename(csv_file, f"{base}_{timestamp}{ext}")

        # Write whatever keys the state dict actually has, not five fixed
        # ones. It used to index message['pan'] and message['tilt'] straight,
        # so the moment get_state stopped reporting the gimbal that is not
        # attached any more, this raised a KeyError once a second.
        fields = list(message.keys())

        # A file that already has rows has a column order, and the first row
        # of it says what that order is. Append to that order, blank for a key
        # this sample happens not to carry.
        header = None
        if os.path.exists(csv_file) and os.path.getsize(csv_file) > 0:
            with open(csv_file, newline='') as file:
                header = next(csv.reader(file), None)

        if header is not None:
            known = header[2:] if header[:2] == ['timestamp', 'subject'] else None
            if known is None or any(key not in known for key in fields):
                # Either this file predates the header row, or I am logging a
                # column it does not have. Appending would leave one file
                # holding two different shapes and nothing could read it, so
                # the old rows go aside under a dated name and this starts
                # clean.
                base, ext = os.path.splitext(csv_file)
                os.rename(csv_file, f"{base}_{timestamp}{ext}")
                header = None
            else:
                fields = known

        with open(csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)

            if header is None:
                writer.writerow(['timestamp', 'subject'] + fields)
            writer.writerow([timestamp, subject] + [message.get(key, '') for key in fields])
            file.flush()