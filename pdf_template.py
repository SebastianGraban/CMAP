#!/usr/bin/python

'''
Sets up a PDF template for a pdf to display all the information about a specific
model of the executed neural network.

Can be used as a template for other pdfs as well.

Author: Sebastian Graban (seg)
Creation Date: 2019-08-13
'''

from fpdf import FPDF

class PDF(FPDF):

    # Header for PDF
    def header(self):
        # Select Arial bold 14
        self.set_font("Arial", "B", 14)
        # Try to write title and footer into the header.
        try:
            self.cell(0, 15, self.title, "B1", 0, 'L')
            self.cell(0,15, self.author, 0, 0, "R")
        except:
            raise IOError("Ensure that title and author is set before creating the header or creating a page")
        self.ln(20)

    # Footer for PDF
    def footer(self):
        # Go to 1.5 cm from bottom
        self.set_y(-15)
        # Select Arial italic 8
        self.set_font('Arial', 'I', 8)
        # Print centered page number
        self.cell(0, 10, 'Page %s' % self.page_no(), 0, 0, 'C')

    # Image along with a caption
    def image_and_desc(self,image_url,w=0,h=0,x=None,y=None,desc=""):
        # Write the image
        self.image(image_url,x,y,w,h)
        # Select Arial 11
        self.set_font("Arial", "",11)
        # Write text for caption under image.
        self.cell(0,5,desc,0,0,"C")
        self.ln(20)
