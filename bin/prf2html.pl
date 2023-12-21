#!/usr/bin/perl

print "<!DOCTYPE html>\n<html lang=\"ja\">\n  <head>\n    <meta charset=\"UTF-8\" />\n";
print "   <meta http-equiv=\"X-UA-Compatible\" content=\"IE=edge\" />\n";
print "    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\" />\n";
print "    <title>sctk outputs</title>\n  </head>\n";
print "    <body>\n";

while(<STDIN>){
    chomp;
    if (/^Speaker\s+sentences/){
	print $_, "</br>\n";
	last;
    }
}

while(<STDIN>){
    chomp;
    if (/^REF:/){
	s/^REF:\s+//;
	s/\s+/\ /g;
	s/\s+$//g;
	@ref = split(/ /, $_);
    }elsif(/^HYP:/){
	s/^HYP:\s+//;
	s/\s+/\ /g;
	s/\s+$//g;
	@hyp = split(/ /, $_);
	if ($#hyp != $#ref){
	    print "error\n";
	    exit(-1)
	}
	@ref_html=("REF:");
	@hyp_html=("HYP:");
	for ($n=0; $n<=$#ref; $n++){
	    #print $ref[$n], " ", $hyp[$n], "\n";
	    if ($ref[$n] eq $hyp[$n]){
		push @ref_html, $ref[$n];
		push @hyp_html, $ref[$n];
	    }elsif ($ref[$n] =~ /^\*/){
		push @ref_html, "<font color=\"green\">".$ref[$n]."</font>";
		push @hyp_html, "<font color=\"green\">".$hyp[$n]."</font>";
	    }elsif ($hyp[$n] =~ /^\*/){
		push @ref_html, "<font color=\"blue\">".$ref[$n]."</font>";
		push @hyp_html, "<font color=\"blue\">".$hyp[$n]."</font>";
	    }else{
		push @ref_html, "<font color=\"red\">".$ref[$n]."</font>";
		push @hyp_html, "<font color=\"red\">".$hyp[$n]."</font>";
	    }
	}
	$html = join(" ", @ref_html);
	print $html, "</br>\n";
	$html = join(" ", @hyp_html);
	print $html, "</br>\n";
	undef @ref_html;
	undef @hyp_html;
	undef @ref;
	undef @hyp;
    }else{
	print $_, "</br>\n";
    }
}
print "    </body>\n</html>\n";
