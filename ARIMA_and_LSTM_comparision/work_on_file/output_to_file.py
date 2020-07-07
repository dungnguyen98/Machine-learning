def result_to_file(filepath, content):
    f = open(filepath, 'w')
    f.write(content)
    f.close()