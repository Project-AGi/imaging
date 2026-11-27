"""Standalone painting: `imagegen paint "a foggy harbor at dawn" -o out.png`.

Also `imagegen check` verifies the engine + weights are in place without
generating.  Everything runs locally via the MossInfer moss-server sidecar.
"""
import argparse
import json
import sys

from .moss import MossClient, MossPaintError


def cmd_paint(args):
    client = MossClient()
    try:
        png = client.generate(args.prompt, width=args.width, height=args.height)
        out = args.output or 'painting.png'
        with open(out, 'wb') as f:
            f.write(png)
        print(f"wrote {out} ({len(png)} bytes)")
    except MossPaintError as error:
        print(f"paint failed: {error}", file=sys.stderr)
        return 1
    finally:
        client.close()
    return 0


def cmd_check(args):
    client = MossClient()
    available = client.available()
    info = {
        'binary': str(client.binary),
        'model_dir': str(client.model_dir),
        'available': available,
    }
    print(json.dumps(info, indent=2))
    return 0 if available else 1


def main():
    ap = argparse.ArgumentParser(description="AIVA's raster painting")
    sub = ap.add_subparsers(required=True)
    paint = sub.add_parser('paint', help='generate one painting')
    paint.add_argument('prompt')
    paint.add_argument('-o', '--output', default=None)
    paint.add_argument('--width', type=int, default=512)
    paint.add_argument('--height', type=int, default=512)
    paint.set_defaults(func=cmd_paint)
    check = sub.add_parser('check', help='verify engine + weights')
    check.set_defaults(func=cmd_check)
    args = ap.parse_args()
    sys.exit(args.func(args))


if __name__ == '__main__':
    main()
