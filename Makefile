.PHONY: test eval plots
test:
	python -m unittest discover -s tests -t . -v
eval:
	python -m coderag eval
plots:
	python -m coderag plots
