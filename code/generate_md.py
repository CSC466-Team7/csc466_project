from traitlets.config import Config
import nbformat as nbf
from nbconvert.exporters import MarkdownExporter
import nbconvert
import os

c = Config()

# Configure our tag removal
c.TagRemovePreprocessor.enabled=True
c.TagRemovePreprocessor.remove_cell_tags = ("hide",)
c.TagRemovePreprocessor.remove_all_outputs_tags = ('hide',)
c.TagRemovePreprocessor.remove_input_tags = ('hide',)
c.preprocessors = ['TagRemovePreprocessor']

# Configure and run out exporter
directory = os.fsencode("notebooks")

for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".ipynb"):
        markdown_prefix = filename.replace(".ipynb", "").replace(" ", "-").lower()
        print(f"Converting {filename} to markdown files with prefix {markdown_prefix}")
        nb_body, resources = MarkdownExporter(config=c).from_filename(f"notebooks/{filename}")
        with open(f'markdown/{markdown_prefix}_cleaned.md', 'w') as writer:
            writer.write(nb_body)