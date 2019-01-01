import java.io.File;
import java.util.Scanner;

public class App {

    public static void main(String[] args) {

        Scanner in = new Scanner(System.in);

        String markdownFilePath;

        while (true) {

            System.out.print("File path : ");

            markdownFilePath = in.nextLine();
            if (new File(markdownFilePath).exists()) break;
            System.out.println("File not exist!");
        }

        TOC.setCatalogueLevel(3);

        String newMarkdownFilePath = markdownFilePath.substring(0, markdownFilePath.length() - 3) + ".gfm.md";

        GFM.convert(markdownFilePath, newMarkdownFilePath);

        System.out.println(newMarkdownFilePath + " has been generated!");
    }
}
