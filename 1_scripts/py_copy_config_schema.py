from asext import asext_ROOT


def append_schema_text(md_file: str, yaml_schema_file: str):
    """Append yaml text to .md file."""
    with open(yaml_schema_file, "r") as f:
        text = f.read()
    schema_text = "\n```yaml\n" + text + "\n```\n"
    schema_text = "\n## Schema:\n" + schema_text
    ### append yaml schema
    with open(md_file, "a") as f:
        f.write(schema_text)
    return


def append_example_config(md_file: str, config_files: list[str]):
    """Append yaml text to .md file."""
    for i, yaml_file in enumerate(config_files):
        with open(yaml_file, "r") as f:
            text = f.read()
        schema_text = "\n```yaml\n" + text + "\n```\n"
        schema_text = f"\n## Example config {i + 1}:\n" + schema_text
        ### append yaml schema
        with open(md_file, "a") as f:
            f.write(schema_text)
    return


#####ANCHOR udpate the blog posts
def main():
    ### Append schema to the .md files
    append_schema_text(
        md_file="./_docs/schema_doc/config_remotes.md",
        yaml_schema_file=f"{asext_ROOT}/jobman/schema/schema_machine.yml",
    )
    ### Append example configuration to the .md files
    append_example_config(
        md_file="./_docs/schema_doc/config_remotes.md",
        config_files=[f"{asext_ROOT}/jobman/schema/sampleConfig_machine_single.yml"],
    )
    return


# if __name__ == "__main__":
#     main()
