"""Fix Pipeline letter formatting in sample .md files.

Adds blank lines around bold headings and before closing lines
so Markdown renders proper paragraph breaks.
"""
import os, re, glob

def fix_letter_format(letter):
    """Add blank lines for proper Markdown paragraph rendering."""
    lines = letter.split('\n')
    result = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        # Add blank line BEFORE a **bold heading** line
        if stripped.startswith('**') and stripped.endswith('**') and result:
            if result[-1].strip() != '':
                result.append('')
        result.append(line)
        # Add blank line AFTER a **bold heading** line
        if stripped.startswith('**') and stripped.endswith('**'):
            result.append('')
    
    text = '\n'.join(result)
    # Fix "Sincerely, Your Care Team" -> separate lines
    text = text.replace('Sincerely, Your Care Team', 'Sincerely,\nYour Care Team')
    # Add blank line before closing lines
    for phrase in ['Thank you for trusting', 'Please feel free to contact',
                   'We are here to support', 'Sincerely,']:
        text = re.sub(r'(?<!\n)\n(' + re.escape(phrase) + ')', r'\n\n\1', text)
    # Clean up triple+ blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text

def fix_file(filepath):
    content = open(filepath).read()
    m = re.search(r'(## Patient Letter[^\n]*\n)', content)
    if not m:
        return False
    header_end = m.end()
    sep = content.find('\n---', header_end)
    if sep < 0:
        return False
    old_letter = content[header_end:sep].strip()
    new_letter = fix_letter_format(old_letter)
    if old_letter == new_letter:
        return False
    new_content = content[:header_end] + '\n' + new_letter + '\n\n' + content[sep:]
    open(filepath, 'w').write(new_content)
    return True

base = os.path.dirname(os.path.abspath(__file__))
fixed = 0
for folder in ['breast_pipeline', 'pdac_pipeline']:
    for f in sorted(glob.glob(os.path.join(base, folder, 'sample_*.md'))):
        if fix_file(f):
            fixed += 1
            print(f'Fixed: {folder}/{os.path.basename(f)}')
        else:
            print(f'  OK: {folder}/{os.path.basename(f)}')
print(f'\nTotal fixed: {fixed}')
