#!/bin/bash
# Post-generation patch for ogen-generated code to handle null values in optional fields

set -e

OGEN_FILE="internal/ogenapi/oas_json_gen.go"

if [ ! -f "$OGEN_FILE" ]; then
    echo "Error: $OGEN_FILE not found"
    exit 1
fi

echo "Patching ogen-generated code to handle null values..."

# Backup original
cp "$OGEN_FILE" "$OGEN_FILE.bak"

# Patch OptString.Decode to handle null values
# The issue: ogen's OptString.Decode() doesn't check for null before trying to decode as string
# Solution: Check jx.Next() type before decoding

cat > /tmp/patch_optstring.txt << 'EOF'
func (o *OptString) Decode(d *jx.Decoder) error {
	if o == nil {
		return errors.New("invalid: unable to decode OptString to nil")
	}
	// Check if the next value is null without consuming it
	switch d.Next() {
	case jx.Null:
		if err := d.Null(); err != nil {
			return err
		}
		// It's null, so leave the field unset
		o.Set = false
		return nil
	case jx.String:
		o.Set = true
		v, err := d.Str()
		if err != nil {
			return err
		}
		o.Value = string(v)
		return nil
	default:
		return errors.New("unexpected json type for OptString")
	}
}
EOF

# Use awk to replace the OptString.Decode function
awk '
/^func \(o \*OptString\) Decode\(d \*jx\.Decoder\) error \{/ {
    system("cat /tmp/patch_optstring.txt")
    skip = 1
    next
}
skip && /^}$/ {
    skip = 0
    next
}
!skip { print }
' "$OGEN_FILE.bak" > "$OGEN_FILE"

rm /tmp/patch_optstring.txt

echo "✓ Patched OptString.Decode to handle null values"
echo "✓ Patch complete"
