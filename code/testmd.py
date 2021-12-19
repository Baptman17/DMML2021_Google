from mdutils.mdutils import MdUtils


mdFile = MdUtils(file_name='DMML2021_Google/README',title='DMML 2021 Project : Detecting the difficulty level of French texts')
mdFile.new_header(level=1, title="Result without Data Cleaning")
mdFile.create_md_file()